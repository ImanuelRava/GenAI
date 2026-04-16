'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { 
  BookOpen, 
  FlaskConical, 
  Atom, 
  Network, 
  MessageSquare,
  Send,
  ChevronRight,
  ChevronLeft,
  Menu,
  X,
  Sparkles,
  Brain,
  Target,
  Zap,
  Maximize2,
  Minimize2
} from 'lucide-react'

// Course content data
const courseModules = [
  {
    id: 'intro',
    title: 'Introduction to AI in Chemistry',
    icon: Brain,
    lessons: [
      { id: 'what-is-ai-chem', title: 'What is AI in Chemistry?', completed: true },
      { id: 'ml-basics', title: 'Machine Learning Basics', completed: true },
      { id: 'data-chem', title: 'Data in Chemistry', completed: false },
    ]
  },
  {
    id: 'nicobot',
    title: 'NiCOBot: Nickel Catalysis',
    icon: FlaskConical,
    lessons: [
      { id: 'nicobot-intro', title: 'Introduction to NiCOBot', completed: false },
      { id: 'nickel-catalysis', title: 'Nickel Catalysis Basics', completed: false },
      { id: 'c-o-activation', title: 'C-O Bond Activation', completed: false },
      { id: 'ligand-design', title: 'Ligand Design', completed: false },
    ]
  },
  {
    id: 'ligands',
    title: 'Redox Active Ligands',
    icon: Atom,
    lessons: [
      { id: 'ligand-intro', title: 'Introduction to Redox Ligands', completed: false },
      { id: 'nitrogen-ligands', title: 'Nitrogen-Based Ligands', completed: false },
      { id: 'electronic-properties', title: 'Electronic Properties', completed: false },
    ]
  },
  {
    id: 'networks',
    title: 'Citation Networks',
    icon: Network,
    lessons: [
      { id: 'network-intro', title: 'Network Analysis', completed: false },
      { id: 'forward-ref', title: 'Forward References', completed: false },
      { id: 'backward-ref', title: 'Backward References', completed: false },
    ]
  },
]

// AI Chat messages
interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

// Content for each lesson
const lessonContent: Record<string, {
  title: string
  content: string[]
  keyPoints: string[]
  interactive?: boolean
}> = {
  'what-is-ai-chem': {
    title: 'What is AI in Chemistry?',
    content: [
      'Artificial Intelligence (AI) has revolutionized the field of chemistry by enabling researchers to analyze vast amounts of data, predict molecular properties, and accelerate drug discovery processes.',
      'Machine learning algorithms can identify patterns in chemical data that would be impossible for humans to detect manually. These patterns help in predicting reaction outcomes, optimizing synthetic routes, and discovering new materials.',
      'In the context of catalysis research, AI models can predict catalyst performance, suggest optimal ligand structures, and even propose entirely new catalytic systems based on learned patterns from experimental data.',
    ],
    keyPoints: [
      'AI enables analysis of large-scale chemical datasets',
      'Machine learning predicts molecular properties and reaction outcomes',
      'AI accelerates drug discovery and materials science',
      'Catalysis research benefits from AI-driven predictions',
    ],
    interactive: true,
  },
  'ml-basics': {
    title: 'Machine Learning Basics',
    content: [
      'Machine Learning (ML) is a subset of AI that enables computers to learn from data without being explicitly programmed. In chemistry, ML models learn relationships between molecular structures and their properties.',
      'Common ML approaches in chemistry include: Supervised Learning (predicting properties from labeled data), Unsupervised Learning (discovering patterns in unlabeled data), and Reinforcement Learning (optimizing through trial and error).',
      'Deep Learning, a subset of ML using neural networks, has shown remarkable success in molecular property prediction, retrosynthesis planning, and protein structure prediction.',
    ],
    keyPoints: [
      'Supervised Learning: Property prediction from labeled data',
      'Unsupervised Learning: Pattern discovery in unlabeled data',
      'Deep Learning: Neural networks for complex predictions',
      'Reinforcement Learning: Optimization through feedback',
    ],
    interactive: true,
  },
  'data-chem': {
    title: 'Data in Chemistry',
    content: [
      'Chemical data comes in many forms: molecular structures (SMILES, InChI), spectroscopic data (NMR, IR, MS), crystallographic data (CIF files), and experimental results (reaction conditions, yields).',
      'Data quality is crucial for AI applications. Clean, well-curated datasets lead to better model performance. This includes handling missing values, removing duplicates, and ensuring consistency.',
      'Modern databases like PubChem, ChEMBL, and the Cambridge Structural Database provide millions of chemical structures and properties for training AI models.',
    ],
    keyPoints: [
      'Molecular representations: SMILES, InChI, molecular graphs',
      'Data quality affects model performance',
      'Public databases provide training data',
      'Standardization ensures reproducibility',
    ],
    interactive: true,
  },
  'nicobot-intro': {
    title: 'Introduction to NiCOBot',
    content: [
      'NiCOBot is an AI-powered tool designed to predict outcomes in nickel-catalyzed cross-coupling reactions. It leverages machine learning trained on experimental data to suggest optimal reaction conditions.',
      'The system analyzes factors such as substrate structure, ligand choice, base, solvent, and temperature to predict reaction success rates and expected yields.',
      'NiCOBot represents a new paradigm in catalysis research, where AI assists chemists in making data-driven decisions about reaction design and optimization.',
    ],
    keyPoints: [
      'AI-powered reaction prediction for Ni catalysis',
      'Analyzes substrate, ligand, and reaction conditions',
      'Data-driven approach to catalyst design',
      'Accelerates experimental planning',
    ],
    interactive: true,
  },
  'nickel-catalysis': {
    title: 'Nickel Catalysis Basics',
    content: [
      'Nickel has emerged as a powerful alternative to palladium in cross-coupling reactions. Its lower cost and unique reactivity profile make it attractive for both academic and industrial applications.',
      'Nickel can access multiple oxidation states (Ni(0), Ni(I), Ni(II), Ni(III)), enabling diverse reaction pathways not accessible with other metals. This includes single-electron transfer processes.',
      'Key advantages of nickel include: activation of inert bonds (C-Cl, C-O, C-N), tolerance to functional groups, and ability to form challenging bonds such as C(sp3)-C(sp3).',
    ],
    keyPoints: [
      'Nickel is cheaper than palladium',
      'Multiple accessible oxidation states',
      'Activates traditionally inert bonds',
      'Enables challenging C(sp3)-C(sp3) couplings',
    ],
    interactive: true,
  },
  'c-o-activation': {
    title: 'C-O Bond Activation',
    content: [
      'C-O bond activation represents a frontier in cross-coupling chemistry. Phenols, aryl ethers, and esters are abundant but underutilized substrates due to the strength of the C-O bond.',
      'Nickel catalysts excel at C-O activation through oxidative addition, enabling coupling of phenol derivatives that would be unreactive with palladium systems.',
      'The NiCOBot project focuses on understanding and predicting these transformations, providing chemists with tools to design new synthetic routes using sustainable C-O activation.',
    ],
    keyPoints: [
      'C-O bonds are strong but abundant',
      'Nickel enables oxidative addition to C-O',
      'Phenols and esters become viable coupling partners',
      'Sustainable alternative to traditional methods',
    ],
    interactive: true,
  },
  'ligand-design': {
    title: 'Ligand Design for Nickel',
    content: [
      'Ligands control the reactivity and selectivity of nickel catalysts. They influence oxidative addition rates, transmetalation barriers, and reductive elimination pathways.',
      'Phosphine ligands (PCy3, PPh3) are common but nitrogen-based ligands (bipyridine, phenanthroline) often provide superior performance in C-O activation.',
      'AI-driven ligand design tools analyze steric and electronic properties to predict optimal ligand-substrate combinations, accelerating catalyst optimization.',
    ],
    keyPoints: [
      'Ligands control catalyst reactivity',
      'Nitrogen ligands excel in C-O activation',
      'Steric and electronic tuning is crucial',
      'AI predicts optimal ligand choices',
    ],
    interactive: true,
  },
}

export default function ImmersivePage() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [currentLesson, setCurrentLesson] = useState('what-is-ai-chem')
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Welcome to ChemAI Learning! I\'m your AI assistant for chemistry and catalysis. Ask me anything about AI in chemistry, nickel catalysis, ligand design, or the NiCOBot project!',
      timestamp: new Date(),
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Find current lesson data
  const getCurrentLessonData = () => {
    for (const courseModule of courseModules) {
      const lesson = courseModule.lessons.find(l => l.id === currentLesson)
      if (lesson) {
        return {
          module: courseModule,
          lesson,
          content: lessonContent[currentLesson]
        }
      }
    }
    return null
  }

  const currentLessonData = getCurrentLessonData()

  // Handle sending messages
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsTyping(true)

    try {
      // Call the AI chat API
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.map(m => ({ role: m.role, content: m.content })),
        }),
      })

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response || data.error || 'I apologize, I could not generate a response. Please try again.',
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className={`min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 ${fullscreen ? 'fixed inset-0 z-50' : ''}`}>
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-sm border-b border-slate-200 shadow-sm">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setMobileSidebarOpen(!mobileSidebarOpen)}
            >
              {mobileSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            <a href="/" className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                <Atom className="h-5 w-5 text-white" />
              </div>
              <div>
                <span className="text-lg font-bold text-slate-800">ChemAI Learning</span>
                <span className="text-xs text-slate-500 block -mt-1">Immersive Experience</span>
              </div>
            </a>
          </div>
          
          <div className="flex items-center gap-3">
            <a href="/" className="hidden md:block">
              <Button variant="ghost" className="text-slate-600">
                ← Back to Home
              </Button>
            </a>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setFullscreen(!fullscreen)}
              className="text-slate-600"
            >
              {fullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
            <Badge variant="secondary" className="bg-emerald-100 text-emerald-700 border-emerald-200">
              <Sparkles className="h-3 w-3 mr-1" />
              AI Powered
            </Badge>
          </div>
        </div>
      </header>

      <div className="flex pt-16">
        {/* Sidebar */}
        <aside className={`
          fixed md:sticky top-16 left-0 z-30
          w-72 h-[calc(100vh-4rem)] 
          bg-white border-r border-slate-200
          transform transition-transform duration-300 ease-in-out
          ${mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
          ${sidebarOpen ? 'md:w-72' : 'md:w-0 md:overflow-hidden'}
        `}>
          <ScrollArea className="h-full">
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wide">Course Content</h2>
                <Button
                  variant="ghost"
                  size="icon"
                  className="hidden md:flex h-6 w-6"
                  onClick={() => setSidebarOpen(false)}
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
              </div>
              
              {/* Progress */}
              <Card className="mb-4 bg-gradient-to-r from-emerald-50 to-teal-50 border-emerald-200">
                <CardContent className="p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-slate-700">Progress</span>
                    <span className="text-sm text-emerald-600 font-semibold">3/14</span>
                  </div>
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div className="bg-gradient-to-r from-emerald-500 to-teal-500 h-2 rounded-full" style={{ width: '21%' }}></div>
                  </div>
                </CardContent>
              </Card>

              {/* Course Modules */}
              <nav className="space-y-2">
                {courseModules.map((courseModule) => (
                  <div key={courseModule.id} className="space-y-1">
                    <button
                      className="w-full flex items-center gap-3 px-3 py-2 text-left rounded-lg hover:bg-slate-100 transition-colors"
                    >
                      <courseModule.icon className="h-5 w-5 text-emerald-600" />
                      <span className="text-sm font-medium text-slate-700">{courseModule.title}</span>
                    </button>
                    <div className="ml-8 space-y-1">
                      {courseModule.lessons.map((lesson) => (
                        <button
                          key={lesson.id}
                          onClick={() => {
                            setCurrentLesson(lesson.id)
                            setMobileSidebarOpen(false)
                          }}
                          className={`
                            w-full flex items-center gap-2 px-3 py-2 text-left rounded-lg transition-all
                            ${currentLesson === lesson.id 
                              ? 'bg-emerald-100 text-emerald-700 border-l-2 border-emerald-500' 
                              : 'hover:bg-slate-50 text-slate-600'}
                          `}
                        >
                          <span className={`w-2 h-2 rounded-full ${lesson.completed ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                          <span className="text-sm">{lesson.title}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </nav>
            </div>
          </ScrollArea>
        </aside>

        {/* Mobile overlay */}
        {mobileSidebarOpen && (
          <div 
            className="fixed inset-0 bg-black/50 z-20 md:hidden"
            onClick={() => setMobileSidebarOpen(false)}
          />
        )}

        {/* Main Content */}
        <main className={`flex-1 min-h-[calc(100vh-4rem)] ${sidebarOpen ? 'md:ml-0' : ''}`}>
          <div className="flex flex-col lg:flex-row h-full">
            {/* Lesson Content */}
            <div className="flex-1 p-6 overflow-y-auto">
              {!sidebarOpen && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="mb-4 hidden md:flex"
                  onClick={() => setSidebarOpen(true)}
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              )}

              {currentLessonData?.content ? (
                <div className="max-w-3xl mx-auto">
                  {/* Lesson Header */}
                  <div className="mb-8">
                    <Badge variant="outline" className="mb-3 border-emerald-300 text-emerald-600">
                      {currentLessonData.module.title}
                    </Badge>
                    <h1 className="text-3xl font-bold text-white mb-4">
                      {currentLessonData.content.title}
                    </h1>
                    <div className="flex items-center gap-4 text-slate-400">
                      <span className="flex items-center gap-1">
                        <Target className="h-4 w-4" />
                        4 Key Points
                      </span>
                      <span className="flex items-center gap-1">
                        <Zap className="h-4 w-4" />
                        Interactive
                      </span>
                    </div>
                  </div>

                  {/* Lesson Content */}
                  <Card className="bg-white/10 backdrop-blur-sm border-white/20 mb-6">
                    <CardContent className="p-6">
                      {currentLessonData.content.content.map((paragraph, idx) => (
                        <p key={idx} className="text-slate-300 leading-relaxed mb-4 last:mb-0">
                          {paragraph}
                        </p>
                      ))}
                    </CardContent>
                  </Card>

                  {/* Key Points */}
                  <Card className="bg-gradient-to-br from-emerald-900/50 to-teal-900/50 border-emerald-500/30 mb-6">
                    <CardHeader>
                      <CardTitle className="text-lg text-emerald-300 flex items-center gap-2">
                        <Target className="h-5 w-5" />
                        Key Takeaways
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-3">
                        {currentLessonData.content.keyPoints.map((point, idx) => (
                          <li key={idx} className="flex items-start gap-3">
                            <span className="w-6 h-6 rounded-full bg-emerald-500/30 flex items-center justify-center text-emerald-300 text-sm font-semibold shrink-0">
                              {idx + 1}
                            </span>
                            <span className="text-slate-300">{point}</span>
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>

                  {/* Interactive Demo */}
                  {currentLessonData.content.interactive && (
                    <Card className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 border-purple-500/30">
                      <CardHeader>
                        <CardTitle className="text-lg text-purple-300 flex items-center gap-2">
                          <Sparkles className="h-5 w-5" />
                          Try It Yourself
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-slate-400 mb-4">
                          Use the AI assistant on the right to ask questions about this topic!
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {['Explain more about this', 'Give me examples', 'How does this relate to NiCOBot?'].map((prompt, idx) => (
                            <Button
                              key={idx}
                              variant="outline"
                              size="sm"
                              className="border-purple-400/50 text-purple-300 hover:bg-purple-500/20"
                              onClick={() => {
                                setInputValue(prompt)
                              }}
                            >
                              {prompt}
                            </Button>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Navigation */}
                  <div className="flex items-center justify-between mt-8 pt-6 border-t border-white/10">
                    <Button variant="ghost" className="text-slate-400">
                      ← Previous Lesson
                    </Button>
                    <Button className="bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600">
                      Next Lesson →
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <BookOpen className="h-16 w-16 text-slate-500 mb-4" />
                  <h2 className="text-2xl font-semibold text-white mb-2">Select a Lesson</h2>
                  <p className="text-slate-400">Choose a topic from the sidebar to begin learning</p>
                </div>
              )}
            </div>

            {/* AI Chat Sidebar */}
            <div className="w-full lg:w-96 border-t lg:border-t-0 lg:border-l border-white/10 bg-slate-900/50 backdrop-blur-sm">
              <div className="flex flex-col h-[50vh] lg:h-[calc(100vh-4rem)]">
                {/* Chat Header */}
                <div className="p-4 border-b border-white/10">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                      <MessageSquare className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">ChemAI Assistant</h3>
                      <span className="text-xs text-emerald-400">Online • Ready to help</span>
                    </div>
                  </div>
                </div>

                {/* Chat Messages */}
                <ScrollArea className="flex-1 p-4" ref={chatContainerRef}>
                  <div className="space-y-4">
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`
                            max-w-[85%] rounded-2xl px-4 py-3
                            ${message.role === 'user'
                              ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white'
                              : 'bg-white/10 backdrop-blur-sm text-slate-200 border border-white/10'}
                          `}
                        >
                          <p className="text-sm leading-relaxed">{message.content}</p>
                          <span className={`text-xs mt-2 block ${message.role === 'user' ? 'text-emerald-100' : 'text-slate-500'}`}>
                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        </div>
                      </div>
                    ))}
                    {isTyping && (
                      <div className="flex justify-start">
                        <div className="bg-white/10 backdrop-blur-sm rounded-2xl px-4 py-3 border border-white/10">
                          <div className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>

                {/* Chat Input */}
                <div className="p-4 border-t border-white/10">
                  <div className="flex gap-2">
                    <Input
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Ask about chemistry..."
                      className="bg-white/5 border-white/10 text-white placeholder:text-slate-500 focus:border-emerald-500"
                    />
                    <Button
                      onClick={handleSendMessage}
                      disabled={!inputValue.trim() || isTyping}
                      className="bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 shrink-0"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {/* Quick Prompts */}
                  <div className="flex flex-wrap gap-2 mt-3">
                    {['Tell me about NiCOBot', 'Explain ligands', 'What is C-O activation?'].map((prompt, idx) => (
                      <Button
                        key={idx}
                        variant="ghost"
                        size="sm"
                        className="text-xs text-slate-400 hover:text-white hover:bg-white/10"
                        onClick={() => setInputValue(prompt)}
                      >
                        {prompt}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
