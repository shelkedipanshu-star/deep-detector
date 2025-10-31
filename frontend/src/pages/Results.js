import React, { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

function ConfidenceBadge({ label, confidence }){
  const color = label==='FAKE' ? 'from-rose-500 to-rose-600' : 'from-emerald-500 to-emerald-600'
  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r ${color} text-white font-semibold`}> 
      <span className="text-sm tracking-wide">{label}</span>
      <span className="text-xs bg-black/20 px-2 py-0.5 rounded-full">{confidence.toFixed(2)}%</span>
    </div>
  )
}

export default function Results(){
  const { id } = useParams()
  const location = useLocation()
  const navigate = useNavigate()
  const [data, setData] = useState(location.state || null)
  const [err, setErr] = useState('')

  useEffect(() => {
    const fetchRes = async () => {
      try {
        const res = await axios.get(`/result/${id}`)
        setData(prev => ({...res.data, ...prev}))
      } catch (e) { setErr('Failed to load result') }
    }
    if (!data) fetchRes()
  }, [id])

  const isVideo = useMemo(()=> Array.isArray(data?.timeline), [data])

  return (
    <div className="relative min-h-screen px-6 py-10 overflow-hidden">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-24 w-[28rem] h-[28rem] rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-[30rem] h-[30rem] rounded-full bg-secondary/20 blur-3xl" />
      </div>
      <div className="max-w-5xl mx-auto">
        <motion.h2 initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="text-3xl font-extrabold">Detection Result</motion.h2>
      {err && <div className="mt-2 text-red-400">{err}</div>}

      {data && (
        <div className="mt-8 grid md:grid-cols-2 gap-8">
          <div className="glass rounded-2xl p-6">
            <div className="text-white/70 text-sm">Verdict</div>
            <div className="mt-3"><ConfidenceBadge label={data.label} confidence={data.confidence} /></div>

            {!isVideo && data.heatmap_path && (
              <div className="mt-6">
                <div className="text-white/70 text-sm mb-2">Grad-CAM Heatmap</div>
                <img className="rounded-xl border border-white/10" alt="heatmap" src={`/uploads/${data.heatmap_path}`} />
              </div>
            )}
          </div>

          <div className="glass rounded-2xl p-6">
            {!isVideo ? (
              <div>
                <div className="text-white/70 text-sm">Confidence</div>
                <div className="mt-3 text-5xl font-extrabold">{data.confidence.toFixed(2)}%</div>
                <div className="mt-2 text-white/60 text-sm">Model certainty based on artifact patterns.</div>
              </div>
            ) : (
              <div>
                <div className="text-white/70 text-sm">Frame-by-frame Confidence</div>
                <div className="mt-3 h-60">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data.timeline.map((v,i)=>({i, v: Math.round(v*10000)/100}))}>
                      <XAxis dataKey="i" hide />
                      <YAxis domain={[0,100]} tick={{fill:'#9ca3af'}} />
                      <Tooltip contentStyle={{background:'#0b1020', border:'1px solid #1f2937', color:'#fff'}} />
                      <Line type="monotone" dataKey="v" stroke="#7C3AED" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 text-white/60 text-sm">Averaged to produce final verdict.</div>
              </div>
            )}

            <div className="mt-8 flex gap-3">
              <button onClick={()=>navigate('/upload')} className="px-5 py-2 rounded-full bg-primary hover:bg-primary/90">Analyze Another</button>
              <button onClick={()=>navigate('/history')} className="px-5 py-2 rounded-full bg-white/10 border border-white/10">View History</button>
              {data?.id && (
                <a href={`/report/${data.id}`} className="px-5 py-2 rounded-full bg-secondary/80 hover:bg-secondary">Download Report</a>
              )}
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  )
}
