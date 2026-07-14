"""
RedCross Chat Blueprint

Provides the ``/api/redcross/chat`` endpoint with RAG-augmented responses
powered by the RedCross reductive-coupling ligand/reaction database.

Enforces 5 strict database-answering rules:
  1. Database use is MANDATORY for factual claims
  2. Explicitly attribute database data ("According to the database, …")
  3. LLM fallback is a LAST RESORT, must be labelled
  4. Comparison queries must cover ALL mentioned classes
  5. DOIs and citations must be included when available
"""

import logging
from typing import Dict, List, Optional

from flask import Blueprint, request, jsonify
from flask_limiter import Limiter

from core.errors import ValidationError, LLMError
from core.utils import sanitize_input

logger = logging.getLogger(__name__)

redcross_bp = Blueprint('redcross', __name__, url_prefix='/api/redcross')

# ---------------------------------------------------------------------------
# 5-rule RAG system prompt
# ---------------------------------------------------------------------------

_REDCROSS_RAG_SYSTEM_PROMPT = """\
You are RedCross Bot, a specialized AI assistant for Reductive Cross-Coupling \
chemistry with a focus on nitrogen-based ligands.

Provide accurate, helpful responses about:
- Nitrogen ligands (bipyridines, phenanthrolines, bisoxazolines, pybox, \
pyridine-oxazolines, pyridine-imidazolines, and related N-donor ligands) and \
their electronic properties
- Reductive cross-coupling reactions (Ni-catalyzed, Fe-catalyzed, Co-catalyzed)
- Ligand electronic descriptors: HOMO, LUMO, HOMO-LUMO gap, electrophilicity \
index (omega), ionization potential, electron affinity, aromaticity indices (HOMA)
- Structure-activity relationships between ligand electronics and coupling performance
- Ligand selection and design strategies for reductive coupling

DATABASE-ANSWERING RULES:
RULE 1 — DATABASE USAGE IS MANDATORY:
You MUST base every factual claim (electronic parameters, redox potentials,
reaction yields, catalytic performance, etc.) on the database context provided
in the user message. Start your answer with database-sourced data and
present it prominently.

RULE 2 — EXPLICITLY ATTRIBUTE DATABASE DATA:
Whenever you use a value, fact, or finding from the database, you MUST
explicitly state that it comes from the database. Use phrasing such as:
  - "According to the database, …"
  - "The database reports a HOMO value of …"
  - "Database records indicate … (DOI: …)"
Do NOT present database-sourced information as if it were your own knowledge.

RULE 3 — FALLBACK TO LLM KNOWLEDGE IS A LAST RESORT:
You may ONLY fall back to your general / training-data knowledge when:
  (a) the specific information is genuinely NOT available in the database context, AND
  (b) you have already exhausted what the database provides.
In that case you MUST explicitly label the information as LLM knowledge,
using phrasing such as:
  - "(Note: the following is based on general knowledge from LLM training, \
not from the database.)"
  - "Beyond the database, from general chemistry knowledge: …"
You may NEVER silently substitute LLM knowledge for database data.

RULE 4 — COMPARISON QUERIES:
When the user compares ligands or reaction types, you MUST discuss ALL classes
mentioned. Use the database data for each class. If the database has data for
one class but not another, present the available data normally and state clearly
for the other: 'The database does not contain entries for [class X].'
Do NOT fabricate 'no data' claims — only say this when the database context
for that class is genuinely empty.

RULE 5 — DOIs AND CITATIONS:
When the database includes DOIs, always cite them. Format: (DOI: 10.xxxx/…).

Keep responses concise but informative. Use proper chemical nomenclature."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _retrieve_rag_context(message: str) -> str:
    """Retrieve RAG context from the RedCross database."""
    try:
        from modules.redcross_rag import get_redcross_rag
        rag = get_redcross_rag()
        context = rag.retrieve_context(message)
        return context.formatted_context
    except Exception as e:
        logger.warning("RedCross RAG context retrieval failed: %s", e)
        return ""


def _build_messages(
    system_prompt: str,
    user_message: str,
    rag_context: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the messages list for the LLM call.

    Injects RAG context into the user message when available.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Replay conversation history (skip last user message — we replace it)
    if history:
        for msg in history:
            if msg.get("role") in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

    # Build user message with RAG context
    if rag_context:
        full_user = (
            f"MANDATORY FIRST SOURCE — The following data comes from the RedCross "
            f"reductive-coupling database and must be your PRIMARY reference:\n\n"
            f"{rag_context}\n\n"
            f"END DATABASE CONTEXT\n\n"
            f"Now answer the user's question using the database data above as your "
            f"primary source. When citing database values, explicitly say "
            f"'According to the database, …'.\n\n"
            f"User question: {user_message}"
        )
    else:
        full_user = user_message

    messages.append({"role": "user", "content": full_user})
    return messages


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@redcross_bp.route('/chat', methods=['POST'])
def chat():
    """RedCross Bot chat endpoint with RAG augmentation."""
    data = request.get_json(silent=True)
    if not data:
        raise ValidationError("No JSON data provided")

    message = data.get('message', '')
    provider = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')
    history = data.get('history')

    if not message or not message.strip():
        raise ValidationError("Message cannot be empty", field="message")

    message = sanitize_input(message, max_length=16000)

    system_prompt = _REDCROSS_RAG_SYSTEM_PROMPT

    # Retrieve RAG context
    rag_context = _retrieve_rag_context(message)

    # Build messages
    messages = _build_messages(system_prompt, message, rag_context, history)

    try:
        from llm.helpers import get_llm_response

        response = get_llm_response(
            system_prompt=system_prompt,
            user_message=message,
            messages=messages,
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=2000,
        )

        if response:
            return jsonify({
                'success': True,
                'response': response,
                'provider': provider or 'default',
                'rag_context_used': bool(rag_context),
            })
        else:
            raise LLMError("No response received from LLM")

    except (KeyboardInterrupt, SystemExit):
        raise
    except LLMError:
        raise
    except Exception as e:
        logger.error("RedCross chat error: %s", e, exc_info=True)
        raise LLMError(f"Error communicating with LLM: {str(e)}")


def register_redcross_blueprint(app, limiter: Limiter):
    """Register RedCross blueprint with rate limiting."""
    app.register_blueprint(redcross_bp)
    limiter.limit("20 per minute")(app.view_functions['redcross.chat'])
    logger.info("[STARTUP] RedCross chat blueprint registered at /api/redcross/chat")