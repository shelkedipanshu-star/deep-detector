import React from 'react'
import { Link, NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function NavBar() {
  return (
    <div className="sticky top-0 z-40 bg-black/40 backdrop-blur border-b border-white/10">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link to="/" className="font-extrabold tracking-tight text-2xl">
          <span className="text-primary">DEEP</span> <span className="text-white/80">Detector</span>
        </Link>
        <div className="flex items-center gap-6">
          <NavLink className={({isActive}) => `hover:text-primary transition ${isActive?'text-primary':'text-white/80'}`} to="/">Home</NavLink>
          <NavLink className={({isActive}) => `hover:text-primary transition ${isActive?'text-primary':'text-white/80'}`} to="/upload">Upload</NavLink>
          <NavLink className={({isActive}) => `hover:text-primary transition ${isActive?'text-primary':'text-white/80'}`} to="/history">History</NavLink>
          <NavLink className={({isActive}) => `hover:text-primary transition ${isActive?'text-primary':'text-white/80'}`} to="/dashboard">Dashboard</NavLink>
          <NavLink className={({isActive}) => `hover:text-primary transition ${isActive?'text-primary':'text-white/80'}`} to="/login">Login</NavLink>
          <a className="hidden md:inline-flex px-4 py-2 rounded-full bg-primary/90 hover:bg-primary transition shadow-lg shadow-primary/30" href="/upload">Try Now</a>
          <button onClick={()=>{ localStorage.removeItem('token'); window.location.href='/' }} className="px-3 py-1 rounded bg-white/10 border border-white/10">Logout</button>
        </div>
      </div>
    </div>
  )
}
