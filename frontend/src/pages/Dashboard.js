import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts'

export default function Dashboard(){
  const [data, setData] = useState(null)
  const [err, setErr] = useState('')

  useEffect(()=>{
    const token = localStorage.getItem('token')
    if(token){ axios.defaults.headers.common['Authorization'] = `Bearer ${token}` }
    const f = async () => {
      try{ const res = await axios.get('/dashboard'); setData(res.data) }catch(e){ setErr('Login required'); }
    }; f()
  },[])

  const COLORS = ['#10b981','#ef4444']
  const dist = data ? Object.entries(data.distribution || {}) : []
  const pieData = dist.map(([k,v])=>({name:k,value:v}))

  return (
    <div className="relative min-h-screen px-6 py-10 overflow-hidden">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-24 w-[28rem] h-[28rem] rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-[30rem] h-[30rem] rounded-full bg-secondary/20 blur-3xl" />
      </div>
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-extrabold">Dashboard</h2>
      {err && <div className="text-rose-400 mt-2">{err}</div>}
      {data && (
        <div className="mt-6 grid md:grid-cols-3 gap-6">
          <div className="glass p-4 rounded-xl"><div className="text-sm text-white/60">Total Detections</div><div className="text-3xl font-bold">{data.total}</div></div>
          <div className="glass p-4 rounded-xl"><div className="text-sm text-white/60">Avg Confidence</div><div className="text-3xl font-bold">{(data.avg_confidence*100).toFixed(1)}%</div></div>
          <div className="glass p-4 rounded-xl">
            <div className="text-sm text-white/60">Distribution</div>
            <div className="h-36">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart><Pie data={pieData} dataKey="value" nameKey="name" outerRadius={50}>
                  {pieData.map((e,i)=>(<Cell key={i} fill={COLORS[i%COLORS.length]} />))}
                </Pie></PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
      {data && (
        <div className="mt-8 glass p-6 rounded-xl border border-white/10">
          <div className="text-sm text-white/60 mb-2">Recent</div>
          <div className="space-y-2">
            {data.recent.map(r=> (
              <div key={r.id} className="flex justify-between text-sm">
                <div className="truncate max-w-md">{r.file_path}</div>
                <div className={`${r.label==='FAKE'?'text-rose-400':'text-emerald-400'}`}>{r.label}</div>
                <div>{(r.confidence*100).toFixed(1)}%</div>
                <div className="text-white/60">{r.timestamp}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      </div>
    </div>
  )
}
