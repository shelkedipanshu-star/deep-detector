import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Landing() {
  return (
    <div className="relative overflow-hidden">
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[40rem] h-[40rem] bg-primary/25 blur-3xl rounded-full -top-40 -left-40 animate-pulse" />
        <div className="absolute w-[50rem] h-[50rem] bg-secondary/15 blur-3xl rounded-full -bottom-40 -right-40 animate-pulse" />
      </div>

      <div className="max-w-7xl mx-auto px-6 py-24 grid md:grid-cols-2 gap-12 items-center">
        <div>
          <motion.h1 initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{duration:0.6}} className="text-5xl md:text-7xl font-extrabold leading-tight">
            Detect Deepfakes with <span className="text-primary">Precision</span>
          </motion.h1>
          <motion.p initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{delay:0.2}} className="mt-6 text-white/80 text-lg md:text-xl">
            Upload an image or video. Get an instant verdict with confidence, Grad-CAM heatmaps, and frame-by-frame analysis.
          </motion.p>
          <motion.div initial={{opacity:0}} animate={{opacity:1}} transition={{delay:0.4}} className="mt-10 flex flex-wrap gap-4">
            <Link to="/upload" className="px-7 py-3 rounded-full bg-primary hover:bg-primary/90 transition shadow-xl shadow-primary/30">Try DIP Detector</Link>
            <a href="/history" className="px-7 py-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/10">View History</a>
          </motion.div>
        </div>

        <motion.div initial={{opacity:0, scale:0.95}} animate={{opacity:1, scale:1}} transition={{duration:0.6}} className="glass rounded-3xl p-6 md:p-8">
          <div className="aspect-video rounded-2xl bg-gradient-to-br from-primary/20 to-secondary/20 border border-white/10 flex items-center justify-center overflow-hidden">
            <motion.div initial={{scale:1.05}} animate={{scale:1}} transition={{duration:3, repeat:Infinity, repeatType:'reverse'}} className="text-center">
              <div className="text-7xl">🧠</div>
              <div className="mt-4 text-white/80">AI-powered Deepfake Detection</div>
            </motion.div>
          </div>
          <div className="mt-6 grid grid-cols-3 gap-4">
            {['Grad-CAM','Timeline','Confidence'].map((t,i)=> (
              <motion.div whileHover={{y:-4}} key={i} className="glass rounded-xl p-4 text-center border-white/10">
                <div className="text-sm text-white/60">{t}</div>
                <div className="mt-1 text-xl font-bold text-primary">Pro</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
