import React from 'react'
import { Routes, Route, Link, useLocation, Navigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Landing from './pages/Landing'
import Upload from './pages/Upload'
import Processing from './pages/Processing'
import Results from './pages/Results'
import History from './pages/History'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import NavBar from './components/NavBar'
import ProtectedRoute from './components/ProtectedRoute'

export default function App() {
  const location = useLocation()
  const authed = !!localStorage.getItem('token')
  const onAuthRoute = location.pathname === '/login' || location.pathname === '/register'
  const showNav = authed && !onAuthRoute
  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-slate-900 to-black text-white">
      {showNav && <NavBar />}
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/" element={authed ? <Landing /> : <Navigate to="/login" replace />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/upload" element={<Upload />} />
            <Route path="/processing/:id" element={<Processing />} />
            <Route path="/results/:id" element={<Results />} />
            <Route path="/history" element={<History />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Route>
        </Routes>
      </AnimatePresence>
    </div>
  )
}
