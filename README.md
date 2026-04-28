# GenAI Research Platform

A comprehensive web-based educational platform for Artificial Intelligence, Transition Metal Catalysis, Virology, and Redox-Active Ligands research.

## Project Structure

```
download/
├── index.html                 # Main landing page
├── README.md                  # Project documentation
│
├── AI/                        # Machine Learning Education
│   ├── index.html            # AI section overview
│   ├── ml-basics.html        # Module 1: Introduction to ML
│   ├── ml-supervised.html    # Module 2: Supervised Learning
│   └── ml-deeplearning.html  # Module 3: Deep Learning
│
├── TMC/                       # Transition Metal Catalysis
│   ├── index.html            # TMC section overview
│   ├── introduction.html     # Introduction page
│   ├── lecture-1-basics.html # Lecture 1: Cross-Coupling Basics
│   ├── lecture-2-metal.html  # Lecture 2: Metal Centers
│   ├── lecture-3-mechanism.html # Lecture 3: Mechanism
│   ├── quiz.html             # Interactive quiz with RDKit
│   ├── citation-tool.html    # Citation network tool
│   ├── data-extraction.html  # Data extraction tool
│   └── tmc-embed.html        # Embedded content
│
├── virus/                     # Virology Section
│   └── index.html            # Virology overview
│
├── redox-ligands/            # Redox-Active Ligands
│   └── index.html            # Redox ligands overview
│
└── backend/                   # Flask Backend
    ├── app.py                # Main Flask application
    ├── requirements.txt      # Python dependencies
    ├── README.md             # Backend documentation
    └── modules/              # Python modules
        ├── DOI.py
        ├── Cross_Reference.py
        ├── Forward_Reference.py
        └── Local_Reference.py
```

## Features

### 1. Machine Learning Education (AI Section)

The AI section provides comprehensive, interactive lessons with sidebar navigation and multiple content panels for each module.

#### Module 1: Introduction to Machine Learning (`ml-basics.html`)
| Section | Content |
|---------|---------|
| Overview | Learning objectives and introduction |
| What is ML? | Traditional programming vs ML, key definitions |
| Types of ML | Supervised, Unsupervised, Reinforcement learning comparison |
| ML Workflow | 6-step process from data collection to deployment |
| Applications | ML in chemistry, drug discovery, materials science |
| Math Foundations | Linear algebra, calculus, probability & statistics |

#### Module 2: Supervised Learning (`ml-supervised.html`)
| Section | Content |
|---------|---------|
| Overview | Supervised learning paradigm |
| Classification | Logistic regression, SVM, kernel trick |
| Regression | Linear, polynomial, Ridge/Lasso/Elastic Net |
| Decision Trees | Gini impurity, Random Forest, Gradient Boosting, AdaBoost |
| Evaluation Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, Cross-validation |
| Overfitting | Bias-variance tradeoff, regularization techniques |

#### Module 3: Deep Learning (`ml-deeplearning.html`)
| Section | Content |
|---------|---------|
| Overview | Neural network architecture visualization |
| Neural Networks | Artificial neurons, activation functions (ReLU, Sigmoid, Softmax) |
| Backpropagation | Chain rule, optimization algorithms (SGD, Momentum, Adam) |
| CNNs | Convolution layers, pooling, famous architectures (ResNet, VGG) |
| RNNs & LSTMs | Sequential data, gating mechanisms, GRU |
| Transformers | Self-attention, multi-head attention, BERT, GPT |
| Graph Neural Networks | Message passing, GCN, GAT, molecular applications |

#### Coming Soon
- Unsupervised Learning (Clustering, PCA, t-SNE, UMAP)
- Natural Language Processing (Tokenization, Embeddings, Fine-tuning)
- ML for Chemistry (Molecular representations, Property prediction)

### 2. Transition Metal Catalysis (TMC Section)
- Interactive lecture content with sidebar navigation
- Cross-coupling reaction mechanisms
- Named reactions (Suzuki, Heck, Stille, etc.)
- Catalytic cycle explanations

### 3. Interactive Quiz System
- 50 questions on chemical reactions
- RDKit-rendered reaction diagrams
- Navigation between questions
- Progress tracking
- Score calculation

### 4. Citation Network Analysis
- Forward citation analysis
- Backward reference tracking
- Cross-reference network building
- Network visualization

## Backend API Endpoints

### Static Pages
| Route | Description |
|-------|-------------|
| `/` | Main landing page |
| `/TMC/` | Transition Metal Catalysis section |
| `/AI/` | Machine Learning section |
| `/virus/` | Virology section |
| `/redox-ligands/` | Redox-Active Ligands section |

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/network` | POST | Citation network analysis |
| `/api/molecules` | GET | Get molecule structures (base64) |
| `/api/reactions` | GET | Get all reaction diagrams |
| `/api/reaction/<key>` | GET | Get single reaction diagram |

### Network Analysis Types
- `forward` - Forward citation network
- `backward` - Backward reference network
- `cross` - Cross-reference network

## Installation

### Prerequisites
- Python 3.8+
- Node.js (optional, for development)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will start at `http://localhost:5000`

### Dependencies
```
Flask>=2.0.0
flask-cors>=3.0.0
networkx>=2.6.0
rdkit>=2021.03.1
```

## Usage

### Running Locally

1. Start the Flask backend:
```bash
cd backend
python app.py
```

2. Open your browser to `http://localhost:5000`

### Quiz Navigation
- Use the question number buttons to jump to any question
- Previous/Next buttons for sequential navigation
- Finish Quiz button to submit and see your score

### Reaction Diagrams
Reaction diagrams are generated using RDKit's `ReactionFromSmarts` and `ReactionToImage` functions. The API returns base64-encoded PNG images.

Supported reactions:
- Suzuki-Miyaura Coupling
- Heck Reaction
- Sonogashira Coupling
- Buchwald-Hartwig Amination
- Stille Coupling
- Negishi Coupling
- Kumada Coupling
- Hiyama Coupling
- Grignard Addition

## Technology Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Flask (Python)
- **Chemistry**: RDKit for molecular visualization
- **Network Analysis**: NetworkX for graph operations

## Design Features

### UI/UX
- Clean white theme with subtle gradients
- Responsive design for mobile devices
- Sidebar navigation for lesson content
- Interactive cards and visual diagrams
- Smooth animations and transitions

### Lesson Page Features
Each ML lesson page includes:
- **Sidebar Navigation**: Quick access to all sections within a module
- **Multiple Content Panels**: Switch between topics without page reload
- **Formula Boxes**: Dark-themed code blocks for mathematical formulas
- **Info Cards**: Visual breakdowns of key concepts
- **Comparison Tables**: Side-by-side algorithm comparisons
- **Step-by-Step Diagrams**: Visual workflow explanations
- **Module Navigation**: Previous/Next buttons between modules

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- RDKit for chemistry visualization capabilities
- NetworkX for graph analysis
- The open-source chemistry and ML communities

## Recent Updates

### AI Section Enhancement
- Added `ml-basics.html` - Introduction to Machine Learning module with 6 sections
- Added `ml-supervised.html` - Supervised Learning module with 6 sections
- Added `ml-deeplearning.html` - Deep Learning module with 7 sections
- Updated `index.html` with clickable course cards linking to modules
- Updated learning path to reflect new module structure

### Lesson Features
- Sidebar navigation with section icons
- Content panels with smooth animations
- Formula boxes with syntax highlighting
- Responsive design for all screen sizes

---

**Powered by GenAI Research | AI in Chemistry Initiative**
