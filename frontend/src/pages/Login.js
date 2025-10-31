import React, { useState } from 'react'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Login(){
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [err, setErr] = useState('')
  const navigate = useNavigate()

  const onSubmit = async () => {
    try{
      const res = await axios.post('/auth/login', { email, password })
      const token = res.data.access_token
      localStorage.setItem('token', token)
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      navigate('/')
    }catch(e){ setErr(e?.response?.data?.error || 'Login failed') }
  }

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background glows */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-24 w-[28rem] h-[28rem] rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-[30rem] h-[30rem] rounded-full bg-secondary/20 blur-3xl" />
      </div>

      <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{duration:0.5}} className="w-full max-w-md">
        <div className="glass rounded-2xl border border-white/10 p-8 shadow-2xl shadow-primary/20">
          <h2 className="text-4xl font-extrabold tracking-tight">Welcome back</h2>
          <p className="text-white/60 mt-2 text-sm">No account? <a className="text-primary underline" href="/register">Register here</a></p>
          <form className="mt-8 space-y-4" onSubmit={(e)=>{e.preventDefault(); onSubmit()}}>
            <div>
              <label className="text-xs text-white/60">Email</label>
              <input className="mt-1 w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary/60" placeholder="you@example.com" value={email} onChange={e=>setEmail(e.target.value)} required />
            </div>
            <div>
              <label className="text-xs text-white/60">Password</label>
              <input className="mt-1 w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary/60" placeholder="••••••••" type="password" value={password} onChange={e=>setPassword(e.target.value)} required />
            </div>
            {err && <div className="text-rose-400 text-sm">{err}</div>}
            <motion.button type="submit" whileTap={{scale:0.98}} className="w-full px-5 py-3 rounded-xl bg-gradient-to-r from-primary to-secondary shadow-lg shadow-primary/30 hover:shadow-primary/40 transition">Login</motion.button>
          </form>
        </div>
      </motion.div>
    </div>
  )
}
