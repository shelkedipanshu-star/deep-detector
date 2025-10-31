import React, { useCallback, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { motion } from 'framer-motion'
import axios from 'axios'

const accept = {
  'image/*': ['.jpg', '.jpeg', '.png', '.bmp'],
  'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
}

export default function Upload() {
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles && acceptedFiles[0]) {
      setFile(acceptedFiles[0])
      setError('')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, multiple: false })

  const handleUpload = async () => {
    if (!file) return setError('Please select a file.')
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await axios.post('/upload', form, { headers: { 'Content-Type': 'multipart/form-data' }})
      const { id, label } = res.data
      navigate(`/results/${id}`, { state: res.data })
    } catch (e) {
      setError(e?.response?.data?.error || 'Upload failed')
    }
  }

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden px-6 py-16">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-24 -left-24 w-[28rem] h-[28rem] rounded-full bg-primary/20 blur-3xl" />
        <div className="absolute -bottom-24 -right-24 w-[30rem] h-[30rem] rounded-full bg-secondary/20 blur-3xl" />
      </div>

      <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{duration:0.5}} className="w-full max-w-3xl">
        <div className="glass rounded-2xl border border-white/10 p-8">
          <motion.h2 initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} className="text-4xl font-extrabold">Upload Image or Video</motion.h2>
          <p className="mt-2 text-white/60">Drag & drop or click to browse. Supported: JPG, PNG, MP4, MOV, MKV.</p>

          <div {...getRootProps()} className={`mt-8 border-2 border-dashed rounded-2xl p-10 text-center transition ${isDragActive? 'border-primary bg-primary/5':'border-white/10 bg-white/5'}`}>
            <input {...getInputProps()} />
            <div className="text-6xl">⬆️</div>
            <div className="mt-2 text-white/80">{isDragActive? 'Drop your file here' : 'Drag & drop here, or click to select'}</div>
            {file && <div className="mt-4 text-primary">Selected: {file.name}</div>}
          </div>

          {error && <div className="mt-4 text-rose-400">{error}</div>}

          <div className="mt-8 flex gap-4">
            <button onClick={handleUpload} className="px-6 py-3 rounded-full bg-primary hover:bg-primary/90 transition shadow-xl shadow-primary/30">Upload & Analyze</button>
            <button onClick={()=>setFile(null)} className="px-6 py-3 rounded-full bg-white/10 hover:bg-white/20 border border-white/10">Reset</button>
          </div>
        </div>
      </motion.div>
    </div>
  )
}
