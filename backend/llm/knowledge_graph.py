"""
Mock knowledge graph data for fallback when LLM is unavailable.
"""

from typing import Dict, Any


def generate_mock_knowledge_graph(topic: str) -> dict:
    topic_lower = topic.lower()

    graphs = {
        'suzuki': {
            "nodes": [
                {"id": "suzuki", "label": "Suzuki-Miyaura Coupling", "type": "reaction",
                 "description": "Pd-catalyzed cross-coupling between organoboron and organic halide"},
                {"id": "palladium", "label": "Palladium Catalyst", "type": "catalyst",
                 "description": "Transition metal catalyst essential for Suzuki reaction"},
                {"id": "boronic", "label": "Organoboron Reagent", "type": "reagent",
                 "description": "R-B(OH)2 nucleophilic partner"},
                {"id": "halide", "label": "Organic Halide", "type": "reagent",
                 "description": "R'-X electrophilic partner"},
                {"id": "oxidative", "label": "Oxidative Addition", "type": "mechanism",
                 "description": "Pd(0) -> Pd(II), inserts into C-X bond"},
                {"id": "transmetalation", "label": "Transmetalation", "type": "mechanism",
                 "description": "Transfer of R group from boron to Pd"},
                {"id": "reductive", "label": "Reductive Elimination", "type": "mechanism",
                 "description": "Pd(II) -> Pd(0), forms C-C bond"},
                {"id": "biaryl", "label": "Biaryl Product", "type": "product",
                 "description": "R-R' coupled product"},
            ],
            "edges": [
                {"source": "suzuki", "target": "palladium", "label": "catalyzed by"},
                {"source": "suzuki", "target": "boronic", "label": "uses"},
                {"source": "suzuki", "target": "halide", "label": "uses"},
                {"source": "palladium", "target": "oxidative", "label": "undergoes"},
                {"source": "oxidative", "target": "transmetalation", "label": "followed by"},
                {"source": "transmetalation", "target": "reductive", "label": "followed by"},
                {"source": "reductive", "target": "biaryl", "label": "produces"},
            ],
        },
        'heck': {
            "nodes": [
                {"id": "heck", "label": "Heck Reaction", "type": "reaction",
                 "description": "Pd-catalyzed coupling of aryl halide with alkene"},
                {"id": "palladium", "label": "Palladium Catalyst", "type": "catalyst",
                 "description": "Pd(0)/Pd(II) catalytic cycle"},
                {"id": "aryl_halide", "label": "Aryl Halide", "type": "reagent",
                 "description": "Ar-X electrophile"},
                {"id": "alkene", "label": "Alkene", "type": "reagent",
                 "description": "C=C nucleophilic partner"},
                {"id": "migratory", "label": "Migratory Insertion", "type": "mechanism",
                 "description": "Alkene inserts into Pd-Ar bond"},
                {"id": "styrene", "label": "Styrene Derivative", "type": "product",
                 "description": "Ar-CH=CH2 type product"},
            ],
            "edges": [
                {"source": "heck", "target": "palladium", "label": "catalyzed by"},
                {"source": "heck", "target": "aryl_halide", "label": "uses"},
                {"source": "heck", "target": "alkene", "label": "uses"},
                {"source": "palladium", "target": "migratory", "label": "undergoes"},
                {"source": "migratory", "target": "styrene", "label": "produces"},
            ],
        },
    }

    default_graph = {
        "nodes": [
            {"id": "cross_coupling", "label": "Cross-Coupling", "type": "reaction",
             "description": "Metal-catalyzed C-C bond formation"},
            {"id": "oxidative_addition", "label": "Oxidative Addition", "type": "mechanism",
             "description": "M(0) -> M(II), inserts into C-X bond"},
            {"id": "transmetalation", "label": "Transmetalation", "type": "mechanism",
             "description": "Exchange of ligands between metals"},
            {"id": "reductive_elimination", "label": "Reductive Elimination", "type": "mechanism",
             "description": "M(II) -> M(0), forms product"},
            {"id": "palladium", "label": "Palladium", "type": "catalyst",
             "description": "Most common cross-coupling catalyst"},
            {"id": "nickel", "label": "Nickel", "type": "catalyst",
             "description": "Cheaper alternative, activates C-Cl/C-O"},
            {"id": "product", "label": "C-C Bond Product", "type": "product",
             "description": "Coupled organic molecule"},
        ],
        "edges": [
            {"source": "cross_coupling", "target": "oxidative_addition", "label": "step 1"},
            {"source": "oxidative_addition", "target": "transmetalation", "label": "step 2"},
            {"source": "transmetalation", "target": "reductive_elimination", "label": "step 3"},
            {"source": "reductive_elimination", "target": "product", "label": "produces"},
            {"source": "cross_coupling", "target": "palladium", "label": "catalyzed by"},
            {"source": "cross_coupling", "target": "nickel", "label": "catalyzed by"},
        ],
    }

    for key, graph in graphs.items():
        if key in topic_lower:
            return graph

    return default_graph
