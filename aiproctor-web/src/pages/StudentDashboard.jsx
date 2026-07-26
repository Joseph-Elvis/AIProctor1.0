import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function StudentDashboard() {
  const navigate = useNavigate()
  const [student, setStudent] = useState(null)

  useEffect(() => {
    const data = sessionStorage.getItem('student')
    if (!data) { navigate('/login'); return }
    setStudent(JSON.parse(data))
  }, [navigate])

  const logout = () => {
    sessionStorage.removeItem('student')
    navigate('/login')
  }

  if (!student) return null

  const statusMessages = {
    pending:   {
      text:  'Your registration is pending admin approval.',
      type:  'info',
      icon:  '⏳'
    },
    approved:  {
      text:  'Your registration has been approved. Please proceed to your exam station.',
      type:  'success',
      icon:  '✓'
    },
    rejected:  {
      text:  'Your registration was not approved. Please contact your institution.',
      type:  'error',
      icon:  '✗'
    },
    in_exam:   {
      text:  'You are currently registered as in an active exam session.',
      type:  'info',
      icon:  '📋'
    },
    completed: {
      text:  'Your exam has been completed. Results will be communicated by your institution.',
      type:  'info',
      icon:  '🎓'
    },
  }

  const statusInfo = statusMessages[student.status] || statusMessages['pending']

  return (
    <div className="page-center">
      <div className="card">
        <div className="nav">
          <div className="logo" style={{ textAlign: 'left', marginBottom: 0 }}>
            <h1 style={{ fontSize: '20px' }}>AIProctor</h1>
          </div>
          <button
            className="btn btn-sm"
            style={{ background: '#334155', color: '#94a3b8', width: 'auto' }}
            onClick={logout}
          >
            Logout
          </button>
        </div>

        {/* Passport + Name */}
        <div style={{ textAlign: 'center', marginBottom: '28px' }}>
          {student.passport_photo && (
            <img
              src={`data:${student.passport_mime};base64,${student.passport_photo}`}
              alt="Passport"
              className="passport-preview"
              style={{ width: '100px', height: '100px', marginBottom: '12px' }}
            />
          )}
          <h2 style={{ fontSize: '20px', fontWeight: 700 }}>
            {student.full_name}
          </h2>
          <p style={{ color: '#64748b', fontSize: '13px', marginTop: '4px' }}>
            {student.registration_number} &nbsp;·&nbsp; {student.department}
          </p>
        </div>

        <hr className="divider" />

        {/* Status */}
        <div style={{ marginBottom: '20px' }}>
          <p style={{
            fontSize: '12px', color: '#64748b',
            textTransform: 'uppercase', letterSpacing: '0.06em',
            marginBottom: '10px', fontWeight: 600
          }}>
            Registration Status
          </p>
          <div className={`msg msg-${statusInfo.type}`}>
            {statusInfo.icon} &nbsp; {statusInfo.text}
          </div>
        </div>

        {/* Details */}
        <div style={{
          background: '#0f172a', borderRadius: '10px',
          padding: '16px 20px'
        }}>
          {[
            ['Exam ID',    student.exam_id    || 'Not yet assigned'],
            ['Department', student.department],
            ['Status',     student.status?.toUpperCase()],
          ].map(([label, value]) => (
            <div key={label} style={{
              display: 'flex', justifyContent: 'space-between',
              padding: '10px 0',
              borderBottom: '1px solid #1e293b',
              fontSize: '13px'
            }}>
              <span style={{ color: '#64748b' }}>{label}</span>
              <span style={{ fontWeight: 600 }}>{value}</span>
            </div>
          ))}
        </div>

        <p style={{
          textAlign: 'center', color: '#475569',
          fontSize: '12px', marginTop: '24px'
        }}>
          Check back here to see your approval status.
          Contact your exam coordinator if you have questions.
        </p>
      </div>
    </div>
  )
}