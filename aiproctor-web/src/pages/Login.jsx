import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import axios from 'axios'

const API = 'http://127.0.0.1:8000'

export default function Login() {
  const navigate = useNavigate()
  const [role, setRole]       = useState('student')
  const [regNum, setRegNum]   = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [msg, setMsg]         = useState(null)

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMsg(null)

    try {
      if (role === 'student') {
        const res = await axios.post(`${API}/students/login`, {
          registration_number: regNum,
          password
        })
        // Save student info to sessionStorage
        sessionStorage.setItem('student', JSON.stringify(res.data))
        navigate('/student/dashboard')

      } else {
        // Admin login
        const res = await axios.post(`${API}/admin/login`, {
          username: regNum,
          password
        })
        sessionStorage.setItem('admin', JSON.stringify(res.data))
        navigate('/admin/dashboard')
      }
    } catch (err) {
      setMsg({
        type: 'error',
        text: err.response?.data?.detail || 'Login failed. Check your credentials.'
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-center">
      <div className="card">
        <div className="logo">
          <h1>ExamProctor</h1>
          <p>Intelligent Examination Monitoring System</p>
        </div>

        {/* Role Tabs */}
        <div className="role-tabs">
          <button
            className={`role-tab ${role === 'student' ? 'active' : ''}`}
            onClick={() => { setRole('student'); setMsg(null) }}
          >
            Student Login
          </button>
          <button
            className={`role-tab ${role === 'admin' ? 'active' : ''}`}
            onClick={() => { setRole('admin'); setMsg(null) }}
          >
            Admin Login
          </button>
        </div>

        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label>
              {role === 'student' ? 'Registration Number' : 'Admin Username'}
            </label>
            <input
              type="text"
              placeholder={
                role === 'student' ? 'e.g. CSC/2021/001' : 'admin'
              }
              value={regNum}
              onChange={e => setRegNum(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        {msg && (
          <div className={`msg msg-${msg.type}`}>
            {msg.text}
          </div>
        )}

        {role === 'student' && (
          <div className="link-text">
            New student?{' '}
            <Link to="/student/register">Register here</Link>
          </div>
        )}
      </div>
    </div>
  )
}