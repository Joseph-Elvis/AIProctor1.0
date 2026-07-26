import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import axios from 'axios'

const API = 'http://127.0.0.1:8000'

export default function StudentRegister() {
  const navigate  = useNavigate()
  const [form, setForm] = useState({
    full_name: '', registration_number: '',
    password: '', confirm_password: '', department: ''
  })
  const [photo, setPhoto]       = useState(null)
  const [preview, setPreview]   = useState(null)
  const [loading, setLoading]   = useState(false)
  const [msg, setMsg]           = useState(null)

  const handleChange = e => {
    setForm({ ...form, [e.target.name]: e.target.value })
  }

  const handlePhoto = e => {
    const file = e.target.files[0]
    if (!file) return
    setPhoto(file)
    const reader = new FileReader()
    reader.onload = ev => setPreview(ev.target.result)
    reader.readAsDataURL(file)
  }

  const handleSubmit = async e => {
    e.preventDefault()
    setMsg(null)

    if (form.password !== form.confirm_password) {
      setMsg({ type: 'error', text: 'Passwords do not match' })
      return
    }
    if (!photo) {
      setMsg({ type: 'error', text: 'Please upload your passport photo' })
      return
    }

    setLoading(true)
    const data = new FormData()
    data.append('full_name',           form.full_name)
    data.append('registration_number', form.registration_number)
    data.append('password',            form.password)
    data.append('department',          form.department)
    data.append('passport_photo',      photo)

    try {
      await axios.post(`${API}/students/register`, data)
      setMsg({
        type: 'success',
        text: 'Registration submitted successfully. Please wait for admin approval.'
      })
      setTimeout(() => navigate('/login'), 3000)
    } catch (err) {
      setMsg({
        type: 'error',
        text: err.response?.data?.detail || 'Registration failed. Try again.'
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-center">
      <div className="card">
        <div className="logo">
          <h1>AIProctor</h1>
          <p>Student Registration</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Full Name</label>
            <input
              name="full_name"
              placeholder="Enter your full name"
              value={form.full_name}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Registration Number</label>
            <input
              name="registration_number"
              placeholder="e.g. CSC/2021/001"
              value={form.registration_number}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Department</label>
            <input
              name="department"
              placeholder="e.g. Computer Science"
              value={form.department}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              name="password"
              placeholder="Create a password"
              value={form.password}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Confirm Password</label>
            <input
              type="password"
              name="confirm_password"
              placeholder="Repeat your password"
              value={form.confirm_password}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>Passport Photo</label>
            <div
              className="photo-upload"
              onClick={() => document.getElementById('photoInput').click()}
            >
              <input
                id="photoInput"
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={handlePhoto}
              />
              {preview
                ? <img src={preview} alt="Preview" />
                : <p>Click to upload your passport photo</p>
              }
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Submitting...' : 'Submit Registration'}
          </button>
        </form>

        {msg && (
          <div className={`msg msg-${msg.type}`}>
            {msg.text}
          </div>
        )}

        <div className="link-text">
          Already registered? <Link to="/login">Login here</Link>
        </div>
      </div>
    </div>
  )
}