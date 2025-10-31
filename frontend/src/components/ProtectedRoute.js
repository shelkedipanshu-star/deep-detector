import React from 'react'
import { Navigate, Outlet, useLocation } from 'react-router-dom'

export default function ProtectedRoute(){
  const token = localStorage.getItem('token')
  const loc = useLocation()
  if(!token){
    return <Navigate to="/login" state={{ from: loc.pathname }} replace />
  }
  return <Outlet />
}