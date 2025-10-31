import React, { useEffect, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Processing() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const t = setInterval(() => {
      setProgress(p => {
        if (p >= 95) return p
        return p + Math.random() * 7
      })
    }, 300)
    // Immediately navigate to results since backend returns result on upload
    const jump = setTimeout(()=> navigate(`/results/${id}`), 1200)
    return () => { clearInterval(t); clearTimeout(jump) }
  }, [id, navigate])

  return (
    <div className="relative min-h-screen flex items-center justify-center px-6 py-24 text-center overflow-hidden">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-24 w-[28rem] h-[28rem] rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-[30rem] h-[30rem] rounded-full bg-secondary/20 blur-3xl" />
      </div>
      <div className="w-full max-w-xl">
        <motion.h3 initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="text-3xl font-bold">Analyzing...</motion.h3>
        <div className="mt-8 w-full h-3 rounded-full bg-white/10 overflow-hidden">
          <motion.div initial={{width:0}} animate={{width: `${progress}%`}} className="h-full bg-gradient-to-r from-primary to-secondary" />
        </div>
        <div className="mt-4 text-white/60">Running model inference and explainability...</div>
      </div>
    </div>
  )
}
