import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import axios from 'axios'

export default function History(){
  const [items, setItems] = useState([])

  useEffect(()=>{
    const f = async () => {
      try {
        const res = await axios.get('/history')
        setItems(res.data.items || [])
      } catch (e) {}
    }
    f()
  }, [])

  return (
    <div className="relative min-h-screen px-6 py-10 overflow-hidden">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-24 w-[28rem] h-[28rem] rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-[30rem] h-[30rem] rounded-full bg-secondary/20 blur-3xl" />
      </div>
      <div className="max-w-4xl mx-auto">
        <motion.h2 initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="text-3xl font-extrabold">History</motion.h2>
      <div className="mt-6 space-y-4">
        {items.map(it => (
          <div key={it.id} className="glass rounded-xl p-4 flex items-center justify-between">
            <div>
              <div className="text-sm text-white/60">{new Date(it.timestamp).toLocaleString()}</div>
              <div className="text-white/90 text-sm truncate max-w-md">{it.file_path}</div>
            </div>
            <div className="text-right">
              <div className={`text-sm font-bold ${it.label==='FAKE'?'text-rose-400':'text-emerald-400'}`}>{it.label}</div>
              <div className="text-xs text-white/60">{(it.confidence*100).toFixed(2)}%</div>
            </div>
          </div>
        ))}
        {items.length===0 && <div className="text-white/60">No history yet.</div>}
      </div>
      </div>
    </div>
  )
}
