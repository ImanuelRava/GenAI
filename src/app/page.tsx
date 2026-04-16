'use client'

import { useEffect, useRef } from 'react'

// Backend data simulation
const backendData = {
  hero: {
    title: "Accelerating Chemical Discovery",
    description: "Exploring the frontiers of Nickel Catalysis and AI-driven ligand design."
  },
  nicobot: {
    sectionTitle: "Nickel Catalyzed C-O Bond Activation (NiCOBot)",
    teaser: "NiCOBot utilizes machine learning to predict outcomes in Nickel-catalyzed reactions. Click this card to access the dedicated research portal, live tools, and detailed methodology."
  },
  nitrogen: {
    sectionTitle: "Redox Active Ligands",
    teaser: "Nitrogen-based ligands play a pivotal role in modulating the electronic and steric environment of metal centers. Click to explore our AI-driven ligand optimization research and interactive database."
  }
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const particlesArrayRef = useRef<Array<{
    x: number
    y: number
    directionX: number
    directionY: number
    size: number
    color: string
    density: number
  }>>([])
  const mouseRef = useRef({ x: undefined as number | undefined, y: undefined as number | undefined })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const PARTICLE_DENSITY = 9000
    const MOUSE_RADIUS = 150
    const CONNECT_DISTANCE = 120

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      initParticles()
    }

    const handleMouseMove = (event: MouseEvent) => {
      mouseRef.current.x = event.x
      mouseRef.current.y = event.y
    }

    const handleMouseOut = () => {
      mouseRef.current.x = undefined
      mouseRef.current.y = undefined
    }

    const initParticles = () => {
      particlesArrayRef.current = []
      const numberOfParticles = Math.floor((canvas.height * canvas.width) / PARTICLE_DENSITY)
      for (let i = 0; i < numberOfParticles; i++) {
        const size = (Math.random() * 2) + 1
        const x = (Math.random() * ((canvas.width - size * 2) - (size * 2))) + size * 2
        const y = (Math.random() * ((canvas.height - size * 2) - (size * 2))) + size * 2
        const directionX = (Math.random() * 0.4) - 0.2
        const directionY = (Math.random() * 0.4) - 0.2
        const color = '#00f2ff'
        particlesArrayRef.current.push({
          x, y, directionX, directionY, size, color,
          density: (Math.random() * 30) + 1
        })
      }
    }

    const drawParticle = (particle: typeof particlesArrayRef.current[0]) => {
      ctx.beginPath()
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2, false)
      ctx.fillStyle = particle.color
      ctx.fill()
    }

    const updateParticle = (particle: typeof particlesArrayRef.current[0]) => {
      if (particle.x > canvas.width || particle.x < 0) {
        particle.directionX = -particle.directionX
      }
      if (particle.y > canvas.height || particle.y < 0) {
        particle.directionY = -particle.directionY
      }

      const mouse = mouseRef.current
      if (mouse.x !== undefined && mouse.y !== undefined) {
        const dx = mouse.x - particle.x
        const dy = mouse.y - particle.y
        const distance = Math.sqrt(dx * dx + dy * dy)

        if (distance < MOUSE_RADIUS) {
          const forceDirectionX = dx / distance
          const forceDirectionY = dy / distance
          const force = (MOUSE_RADIUS - distance) / MOUSE_RADIUS

          const directionX = forceDirectionX * force * particle.density
          const directionY = forceDirectionY * force * particle.density

          particle.x -= directionX * 5
          particle.y -= directionY * 5
        } else {
          particle.x += particle.directionX
          particle.y += particle.directionY
        }
      } else {
        particle.x += particle.directionX
        particle.y += particle.directionY
      }
      drawParticle(particle)
    }

    const connect = () => {
      const particles = particlesArrayRef.current
      for (let a = 0; a < particles.length; a++) {
        for (let b = a; b < particles.length; b++) {
          const distance = ((particles[a].x - particles[b].x) ** 2) +
            ((particles[a].y - particles[b].y) ** 2)

          if (distance < (CONNECT_DISTANCE * CONNECT_DISTANCE)) {
            const opacityValue = 1 - (distance / (CONNECT_DISTANCE * CONNECT_DISTANCE))
            ctx.strokeStyle = `rgba(112, 0, 255, ${opacityValue})`
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(particles[a].x, particles[a].y)
            ctx.lineTo(particles[b].x, particles[b].y)
            ctx.stroke()
          }
        }
      }
    }

    const animate = () => {
      requestAnimationFrame(animate)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      for (let i = 0; i < particlesArrayRef.current.length; i++) {
        updateParticle(particlesArrayRef.current[i])
      }
      connect()
    }

    // Initialize
    resizeCanvas()
    animate()

    // Event listeners
    window.addEventListener('resize', resizeCanvas)
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseout', handleMouseOut)

    // Cleanup
    return () => {
      window.removeEventListener('resize', resizeCanvas)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseout', handleMouseOut)
    }
  }, [])

  return (
    <div className="min-h-screen bg-[#050a14] text-white overflow-x-hidden" style={{ lineHeight: 1.7 }}>
      {/* Background Canvas */}
      <canvas
        ref={canvasRef}
        className="fixed top-0 left-0 w-full h-full -z-10"
        style={{
          background: 'radial-gradient(circle at center, transparent 0%, #050a14 95%)'
        }}
      />

      {/* Header */}
      <header className="sticky top-0 z-50 py-5 border-b border-white/10 bg-[rgba(5,10,20,0.85)] backdrop-blur-sm">
        <div className="max-w-[1200px] mx-auto px-6 flex justify-between items-center">
          <a href="/" className="text-2xl font-bold bg-gradient-to-r from-[#00f2ff] to-[#7000ff] bg-clip-text text-transparent tracking-wider">
            ChemAI Research
          </a>
          <nav>
            <ul className="flex list-none gap-8">
              <li><a href="/" className="text-white font-medium text-xs uppercase tracking-wider hover:text-[#00f2ff] transition-colors">Home</a></li>
              <li><a href="/immersive" className="text-white font-medium text-xs uppercase tracking-wider hover:text-[#00f2ff] transition-colors">Immersive Learning</a></li>
              <li><a href="/nicobot" className="text-white font-medium text-xs uppercase tracking-wider hover:text-[#00f2ff] transition-colors">NiCOBot</a></li>
              <li><a href="/redox-ligands" className="text-white font-medium text-xs uppercase tracking-wider hover:text-[#00f2ff] transition-colors">Redox Active Ligands</a></li>
            </ul>
          </nav>
        </div>
      </header>

      <main>
        {/* Hero Section */}
        <section className="min-h-[70vh] flex flex-col justify-center items-start py-20">
          <div className="max-w-[1200px] mx-auto px-6">
            <h1 className="text-5xl md:text-6xl font-bold leading-tight mb-5" style={{ textShadow: '0 0 30px rgba(0, 242, 255, 0.2)' }}>
              {backendData.hero.title}
            </h1>
            <div className="text-xl text-[#b0c4de] max-w-[600px] border-l-4 border-[#00ff88] pl-4">
              {backendData.hero.description}
            </div>
            {/* Immersive Learning CTA */}
            <div className="mt-8">
              <a 
                href="/immersive" 
                className="inline-flex items-center gap-2 bg-gradient-to-r from-[#00f2ff] to-[#7000ff] text-white font-semibold px-6 py-3 rounded-full hover:opacity-90 transition-opacity"
              >
                <span>Start Immersive Learning</span>
                <span>→</span>
              </a>
            </div>
          </div>
        </section>

        {/* NiCOBot Section */}
        <section className="py-20 border-b border-white/5">
          <div className="max-w-[1200px] mx-auto px-6">
            <a href="/nicobot" className="block bg-[rgba(10,20,40,0.75)] backdrop-blur-xl border border-white/10 rounded-2xl p-10 mb-8 shadow-[0_10px_40px_rgba(0,0,0,0.4)] transition-all duration-300 hover:-translate-y-1 hover:border-[#00f2ff] hover:shadow-[0_15px_50px_rgba(0,242,255,0.15)] relative group">
              <span className="absolute top-5 right-5 text-xs text-[#00f2ff] border border-[#00f2ff] px-3 py-1.5 rounded-full opacity-70 group-hover:opacity-100 transition-opacity">
                View Project →
              </span>
              <h2 className="text-3xl mb-6 pb-1 border-b border-[#7000ff] inline-block">
                {backendData.nicobot.sectionTitle}
              </h2>
              <div className="grid md:grid-cols-2 gap-10">
                <div>
                  <p className="text-[#b0c4de] mb-4">{backendData.nicobot.teaser}</p>
                  <h3 className="text-[#00f2ff] text-xl mt-5 mb-2">Project Contents:</h3>
                  <ul className="list-none p-0 mt-4 space-y-2">
                    {['Introduction to Transition Metal Catalysis', 'Citation Network Tools', 'Data Extraction Pipeline', 'Live NiCOBot Interface'].map((item, idx) => (
                      <li key={idx} className="flex items-center text-[#b0c4de]">
                        <span className="text-[#00ff88] text-xs mr-3">►</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-gradient-to-br from-[rgba(112,0,255,0.1)] to-transparent rounded-xl p-5 flex items-center justify-center border border-dashed border-[#7000ff] min-h-[250px]">
                  <div className="text-center">
                    <div className="text-6xl font-bold text-[#00f2ff] animate-pulse" style={{ filter: 'drop-shadow(0 0 10px #00f2ff)' }}>
                      Ni
                    </div>
                    <div className="text-[#b0c4de] mt-2">Ni-Catalyst</div>
                    <div className="flex justify-center gap-2.5 mt-5">
                      <div className="w-5 h-5 rounded-full bg-[#00f2ff] shadow-[0_0_10px_#00f2ff]"></div>
                      <div className="w-5 h-5 rounded-full bg-[#7000ff] shadow-[0_0_10px_#7000ff]"></div>
                      <div className="w-5 h-5 rounded-full bg-[#b0c4de]"></div>
                    </div>
                  </div>
                </div>
              </div>
            </a>
          </div>
        </section>

        {/* Nitrogen Ligands Section */}
        <section className="py-20 border-b border-white/5">
          <div className="max-w-[1200px] mx-auto px-6">
            <a href="/redox-ligands" className="block bg-[rgba(10,20,40,0.75)] backdrop-blur-xl border border-white/10 rounded-2xl p-10 mb-8 shadow-[0_10px_40px_rgba(0,0,0,0.4)] transition-all duration-300 hover:-translate-y-1 hover:border-[#00f2ff] hover:shadow-[0_15px_50px_rgba(0,242,255,0.15)] relative group">
              <span className="absolute top-5 right-5 text-xs text-[#00f2ff] border border-[#00f2ff] px-3 py-1.5 rounded-full opacity-70 group-hover:opacity-100 transition-opacity">
                View Project →
              </span>
              <h2 className="text-3xl mb-6 pb-1 border-b border-[#7000ff] inline-block">
                {backendData.nitrogen.sectionTitle}
              </h2>
              <div className="grid md:grid-cols-2 gap-10">
                <div>
                  <p className="text-[#b0c4de] mb-4">{backendData.nitrogen.teaser}</p>
                  <h3 className="text-[#00f2ff] text-xl mt-5 mb-2">Project Contents:</h3>
                  <ul className="list-none p-0 mt-4 space-y-2">
                    {['Nitrogen Ligand Overview', 'AI-Driven Ligand Optimization', 'Quantum Chemical Descriptors', 'Interactive Ligand Database'].map((item, idx) => (
                      <li key={idx} className="flex items-center text-[#b0c4de]">
                        <span className="text-[#00ff88] text-xs mr-3">►</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="bg-gradient-to-br from-[rgba(112,0,255,0.1)] to-transparent rounded-xl p-5 flex items-center justify-center border border-dashed border-[#7000ff] min-h-[250px]">
                  <div className="text-center">
                    <div className="text-6xl font-bold text-[#7000ff] animate-pulse" style={{ filter: 'drop-shadow(0 0 10px #7000ff)' }}>
                      N
                    </div>
                    <div className="text-[#b0c4de] mt-2">N-Donor Ligand</div>
                    <div className="flex justify-center gap-2.5 mt-5">
                      <div className="w-5 h-5 rounded-full bg-[#00f2ff] shadow-[0_0_10px_#00f2ff]"></div>
                      <div className="w-5 h-5 rounded-full bg-[#7000ff] shadow-[0_0_10px_#7000ff]"></div>
                      <div className="w-5 h-5 rounded-full bg-[#b0c4de]"></div>
                    </div>
                  </div>
                </div>
              </div>
            </a>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="text-center py-10 text-[#b0c4de] border-t border-white/10">
        <div className="max-w-[1200px] mx-auto px-6">
          <p>Powered by Python Backend Simulation | AI in Chemistry Initiative</p>
        </div>
      </footer>
    </div>
  )
}
