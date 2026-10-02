import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const API    = 'http://127.0.0.1:8000'
const WS_URL = 'ws://127.0.0.1:8000/ws/alerts'

function calcScore(alerts) {
  if (!alerts || alerts.length === 0) return 100
  let score = 100
  alerts.forEach(a => {
    const type = (a.violation_type || '').toUpperCase()
    if      (type.includes('CAMERA BLOCKED'))    score -= 20
    else if (type.includes('IMPERSONATION'))     score -= 20
    else if (type.includes('MULTIPLE FACES'))    score -= 15
    else if (type.includes('UNAUTHORIZED'))      score -= 15
    else if (type.includes('OBJECT_DETECTED'))   score -= 10
    else if (type.includes('LOOKING'))           score -= 5
    else if (type.includes('NO FACE'))           score -= 5
  })
  return Math.max(0, score)
}

function scoreClass(score) {
  if (score >= 90) return 'score-high'
  if (score >= 70) return 'score-moderate'
  if (score >= 50) return 'score-flagged'
  return 'score-fail'
}

function scoreLabel(score) {
  if (score >= 90) return 'PASS — High Integrity'
  if (score >= 70) return 'PASS — Moderate Integrity'
  if (score >= 50) return 'FLAGGED — Review Required'
  return 'FAIL — Integrity Compromised'
}

function violationColour(type) {
  const t = (type || '').toUpperCase()
  if (t.includes('CAMERA BLOCKED'))  return '#dc2626'
  if (t.includes('MULTIPLE FACES'))  return '#ef4444'
  if (t.includes('IMPERSONATION'))   return '#ef4444'
  if (t.includes('UNAUTHORIZED'))    return '#f97316'
  if (t.includes('OBJECT_DETECTED')) return '#eab308'
  if (t.includes('NO FACE'))         return '#f97316'
  if (t.includes('LOOKING'))         return '#38bdf8'
  return '#94a3b8'
}

// ── Evidence Image Popup ──────────────────────────────────────────────────
function EvidencePopup({ alert, onClose }) {
  if (!alert) return null
  return (
    <div style={{
      position:        'fixed',
      top:             0, left: 0,
      width:           '100vw', height: '100vh',
      background:      'rgba(0,0,0,0.85)',
      display:         'flex',
      alignItems:      'center',
      justifyContent:  'center',
      zIndex:          1000,
    }} onClick={onClose}>
      <div style={{
        background:    '#1e293b',
        borderRadius:  '16px',
        padding:       '24px',
        maxWidth:      '600px',
        width:         '90%',
        border:        `2px solid ${violationColour(alert.violation_type)}`,
      }} onClick={e => e.stopPropagation()}>

        <div style={{
          display:        'flex',
          justifyContent: 'space-between',
          alignItems:     'center',
          marginBottom:   '16px'
        }}>
          <h3 style={{ color: '#e2e8f0', fontSize: '16px', margin: 0 }}>
            Evidence Frame
          </h3>
          <button onClick={onClose} style={{
            background: 'none', border: 'none',
            color: '#64748b', fontSize: '20px', cursor: 'pointer'
          }}>✕</button>
        </div>

        {alert.evidence_frame ? (
          <img
            src={`data:image/jpeg;base64,${alert.evidence_frame}`}
            alt="Evidence"
            style={{
              width:        '100%',
              borderRadius: '8px',
              border:       '1px solid #334155'
            }}
          />
        ) : (
          <div style={{
            background:    '#0f172a',
            borderRadius:  '8px',
            padding:       '40px',
            textAlign:     'center',
            color:         '#64748b'
          }}>
            No evidence frame captured for this alert
          </div>
        )}

        <div style={{ marginTop: '16px' }}>
          {[
            ['Violation',  alert.violation_type],
            ['Candidate',  alert.candidate_id],
            ['Station',    alert.station_id],
            ['Time',       new Date(alert.timestamp).toLocaleString()],
            ['Objects',    alert.objects?.join(', ') || '—'],
          ].map(([label, value]) => (
            <div key={label} style={{
              display:       'flex',
              gap:           '12px',
              padding:       '8px 0',
              borderBottom:  '1px solid #334155',
              fontSize:      '13px'
            }}>
              <span style={{ color: '#64748b', width: '80px' }}>
                {label}
              </span>
              <span style={{
                color:      violationColour(alert.violation_type),
                fontWeight: 600
              }}>
                {value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Report Modal ──────────────────────────────────────────────────────────
function ReportModal({ student, alerts, onClose }) {
  if (!student) return null

  const sAlerts = alerts.filter(
    a => a.candidate_id === student.registration_number
  )
  const score = calcScore(sAlerts)
  const types = {}
  sAlerts.forEach(a => {
    const t = a.violation_type || 'Unknown'
    types[t] = (types[t] || 0) + 1
  })

  const deductionMap = {
    'IMPERSONATION':   20,
    'UNAUTHORIZED':    15,
    'OBJECT_DETECTED': 10,
    'LOOKING':          5,
    'NO FACE':          5,
  }

  return (
    <div style={{
      position:       'fixed',
      top:            0, left: 0,
      width:          '100vw', height: '100vh',
      background:     'rgba(0,0,0,0.85)',
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'center',
      zIndex:         1000,
      overflowY:      'auto',
    }} onClick={onClose}>
      <div style={{
        background:   '#0f172a',
        borderRadius: '16px',
        padding:      '36px',
        maxWidth:     '720px',
        width:        '95%',
        margin:       '20px auto',
        border:       '1px solid #334155',
      }} onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div style={{
          textAlign:    'center',
          marginBottom: '24px',
          paddingBottom:'20px',
          borderBottom: '2px solid #38bdf8'
        }}>
          <h1 style={{
            color: '#38bdf8', fontSize: '24px',
            fontWeight: 800, margin: '0 0 4px'
          }}>
            ExamProctor
          </h1>
          <p style={{ color: '#64748b', fontSize: '13px', margin: 0 }}>
            Examination Integrity Report
          </p>
        </div>

        {/* Student Info */}
        <div style={{
          display:             'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap:                 '1px',
          background:          '#334155',
          borderRadius:        '10px',
          overflow:            'hidden',
          marginBottom:        '20px'
        }}>
          {[
            ['STUDENT NAME',        student.full_name],
            ['REGISTRATION NUMBER', student.registration_number],
            ['DEPARTMENT',          student.department],
            ['EXAM ID',             student.exam_id || '—'],
            ['STATUS',              (student.status || '').toUpperCase()],
            ['EXAM DATE',           new Date().toLocaleDateString()],
          ].map(([label, value]) => (
            <div key={label} style={{
              background: '#1e293b',
              padding:    '12px 16px'
            }}>
              <div style={{
                fontSize: '10px', color: '#64748b',
                fontWeight: 600, marginBottom: '4px',
                textTransform: 'uppercase', letterSpacing: '0.05em'
              }}>
                {label}
              </div>
              <div style={{
                fontSize: '13px', color: '#e2e8f0', fontWeight: 600
              }}>
                {value}
              </div>
            </div>
          ))}
        </div>

        {/* Score */}
        <div style={{
          display:             'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap:                 '12px',
          marginBottom:        '20px'
        }}>
          {[
            ['INTEGRITY SCORE', `${score}%`,          scoreClass(score)],
            ['GRADE',           scoreLabel(score),     scoreClass(score)],
            ['TOTAL VIOLATIONS', sAlerts.length,       'score-moderate'],
          ].map(([label, value, cls]) => (
            <div key={label} style={{
              background:  '#1e293b',
              borderRadius:'10px',
              padding:     '16px',
              textAlign:   'center',
              border:      '1px solid #334155'
            }}>
              <div style={{
                fontSize: '10px', color: '#64748b',
                fontWeight: 600, marginBottom: '8px',
                textTransform: 'uppercase'
              }}>
                {label}
              </div>
              <div className={cls} style={{ fontSize: '20px', fontWeight: 800 }}>
                {value}
              </div>
            </div>
          ))}
        </div>

        {/* Violation Breakdown */}
        <div style={{ marginBottom: '20px' }}>
          <h3 style={{
            fontSize: '12px', color: '#64748b',
            textTransform: 'uppercase', letterSpacing: '0.05em',
            marginBottom: '10px', fontWeight: 600
          }}>
            Violation Breakdown
          </h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Violation Type', 'Count', 'Deduction', 'Total'].map(h => (
                  <th key={h} style={{
                    padding: '10px 12px', textAlign: 'left',
                    fontSize: '11px', color: '#38bdf8',
                    borderBottom: '1px solid #334155',
                    textTransform: 'uppercase'
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(types).map(([t, c]) => {
                let ded = 0
                const tu = t.toUpperCase()
                for (const [key, val] of Object.entries(deductionMap)) {
                  if (tu.includes(key)) { ded = val; break }
                }
                return (
                  <tr key={t}>
                    <td style={{
                      padding: '10px 12px', fontSize: '12px',
                      color: violationColour(t),
                      borderBottom: '1px solid #1e293b'
                    }}>{t}</td>
                    <td style={{
                      padding: '10px 12px', fontSize: '12px',
                      color: '#e2e8f0', textAlign: 'center',
                      borderBottom: '1px solid #1e293b'
                    }}>{c}</td>
                    <td style={{
                      padding: '10px 12px', fontSize: '12px',
                      color: '#ef4444', textAlign: 'center',
                      borderBottom: '1px solid #1e293b'
                    }}>-{ded} pts</td>
                    <td style={{
                      padding: '10px 12px', fontSize: '12px',
                      color: '#ef4444', textAlign: 'center',
                      borderBottom: '1px solid #1e293b'
                    }}>-{Math.min(ded * c, 100)} pts</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        {/* Violation Timeline */}
        {sAlerts.length > 0 && (
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{
              fontSize: '12px', color: '#64748b',
              textTransform: 'uppercase', letterSpacing: '0.05em',
              marginBottom: '10px', fontWeight: 600
            }}>
              Violation Timeline
            </h3>
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {['#', 'Time', 'Violation', 'Objects'].map(h => (
                      <th key={h} style={{
                        padding: '8px 12px', textAlign: 'left',
                        fontSize: '11px', color: '#38bdf8',
                        borderBottom: '1px solid #334155',
                        position: 'sticky', top: 0,
                        background: '#0f172a'
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sAlerts
                    .filter(a => !a.violation_type?.includes('EXAM_ENDED'))
                    .map((a, i) => (
                      <tr key={a._id}>
                        <td style={{
                          padding: '8px 12px', fontSize: '11px',
                          color: '#64748b',
                          borderBottom: '1px solid #1e293b'
                        }}>{i + 1}</td>
                        <td style={{
                          padding: '8px 12px', fontSize: '11px',
                          color: '#64748b',
                          borderBottom: '1px solid #1e293b'
                        }}>
                          {new Date(a.timestamp).toLocaleTimeString()}
                        </td>
                        <td style={{
                          padding: '8px 12px', fontSize: '11px',
                          color: violationColour(a.violation_type),
                          fontWeight: 600,
                          borderBottom: '1px solid #1e293b'
                        }}>{a.violation_type}</td>
                        <td style={{
                          padding: '8px 12px', fontSize: '11px',
                          color: '#94a3b8',
                          borderBottom: '1px solid #1e293b'
                        }}>
                          {a.objects?.join(', ') || '—'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Actions */}
        <div style={{
          display:        'flex',
          justifyContent: 'space-between',
          alignItems:     'center',
          paddingTop:     '16px',
          borderTop:      '1px solid #334155'
        }}>
          <p style={{ color: '#334155', fontSize: '11px', margin: 0 }}>
            Generated by ExamProctor Examination Monitoring System
          </p>
          <div style={{ display: 'flex', gap: '10px' }}>
            <a
              href={`${API}/reports/${student.registration_number}`}
              target="_blank"
              rel="noreferrer"
              style={{
                padding:        '10px 20px',
                background:     '#1e3a5f',
                color:          '#7dd3fc',
                borderRadius:   '8px',
                fontSize:       '13px',
                fontWeight:     700,
                textDecoration: 'none',
              }}
            >
              ↓ Download PDF
            </a>
            <button onClick={onClose} style={{
              padding:      '10px 20px',
              background:   '#334155',
              color:        '#94a3b8',
              border:       'none',
              borderRadius: '8px',
              fontSize:     '13px',
              fontWeight:   700,
              cursor:       'pointer'
            }}>
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}


// ── Delete Student Modal ────────────────────────────────────────────────
function DeleteStudentModal({ student, onClose, onConfirm }) {
  const [note, setNote]         = useState('')
  const [sending, setSending]   = useState(false)
  const [error, setError]       = useState(null)

  if (!student) return null

  const handleConfirm = async () => {
    if (!note.trim()) {
      setError('Please add a note for the student before deleting.')
      return
    }
    setError(null)
    setSending(true)
    try {
      await onConfirm(student.registration_number, note.trim())
    } catch (err) {
      setError(
        err.response?.data?.detail || 'Failed to delete student.'
      )
      setSending(false)
    }
  }

  return (
    <div style={{
      position:       'fixed',
      top:            0, left: 0,
      width:          '100vw', height: '100vh',
      background:     'rgba(0,0,0,0.85)',
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'center',
      zIndex:         1000,
    }} onClick={sending ? undefined : onClose}>
      <div style={{
        background:   '#1e293b',
        borderRadius: '16px',
        padding:      '28px',
        maxWidth:     '480px',
        width:        '90%',
        border:       '1px solid #450a0a',
      }} onClick={e => e.stopPropagation()}>

        <h3 style={{ color: '#fca5a5', fontSize: '17px', margin: '0 0 6px' }}>
          Delete Student
        </h3>
        <p style={{ color: '#94a3b8', fontSize: '13px', margin: '0 0 20px' }}>
          You're about to permanently remove{' '}
          <strong style={{ color: '#e2e8f0' }}>{student.full_name}</strong>
          {' '}({student.registration_number}). Before this is completed,
          add a note explaining why — the student will see this note
          if they try to log in again, so it should clarify what
          happened and what to do next (e.g. re-register, contact
          the department, etc.).
        </p>

        <textarea
          value={note}
          onChange={e => setNote(e.target.value)}
          placeholder="e.g. Your registration was removed due to a duplicate entry. Please re-register with your correct department selected."
          rows={4}
          style={{
            width:        '100%',
            resize:       'vertical',
            background:   '#0f172a',
            border:       '1px solid #334155',
            borderRadius: '8px',
            color:        '#e2e8f0',
            fontSize:     '13px',
            padding:      '10px 12px',
            fontFamily:   'inherit',
            boxSizing:    'border-box',
          }}
        />

        {error && (
          <div className="msg msg-error" style={{ marginTop: '12px' }}>
            {error}
          </div>
        )}

        <div style={{
          display:        'flex',
          justifyContent: 'flex-end',
          gap:            '10px',
          marginTop:      '20px'
        }}>
          <button
            onClick={onClose}
            disabled={sending}
            style={{
              padding:      '10px 18px',
              background:   '#334155',
              color:        '#94a3b8',
              border:       'none',
              borderRadius: '8px',
              fontSize:     '13px',
              fontWeight:   700,
              cursor:       sending ? 'default' : 'pointer'
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={sending}
            style={{
              padding:      '10px 18px',
              background:   '#450a0a',
              color:        '#fca5a5',
              border:       'none',
              borderRadius: '8px',
              fontSize:     '13px',
              fontWeight:   700,
              cursor:       sending ? 'default' : 'pointer',
              opacity:      sending ? 0.7 : 1,
            }}
          >
            {sending ? 'Sending note & deleting...' : 'Send Note & Delete'}
          </button>
        </div>
      </div>
    </div>
  )
}


// ── Main Admin Dashboard ──────────────────────────────────────────────────
export default function AdminDashboard() {
  const navigate = useNavigate()
  const ws       = useRef(null)

  const [students,       setStudents]       = useState([])
  const [alerts,         setAlerts]         = useState([])
  const [liveAlerts,     setLiveAlerts]     = useState([])
  const [view,           setView]           = useState('students')
  const [loading,        setLoading]        = useState(true)
  const [msg,            setMsg]            = useState(null)
  const [wsStatus,       setWsStatus]       = useState('connecting')
  const [evidenceAlert,  setEvidenceAlert]  = useState(null)
  const [reportStudent,  setReportStudent]  = useState(null)
  const [deleteTarget,   setDeleteTarget]   = useState(null)

  useEffect(() => {
    const admin = sessionStorage.getItem('admin')
    if (!admin) { navigate('/login'); return }
    fetchAll()
    connectWebSocket()
    return () => { if (ws.current) ws.current.close() }
  }, [navigate])

  const fetchAll = async () => {
    setLoading(true)
    try {
      const [s, a] = await Promise.all([
        axios.get(`${API}/students`),
        axios.get(`${API}/alerts`)
      ])
      setStudents(s.data.students || [])
      setAlerts(a.data.alerts     || [])
    } catch {
      setMsg({ type: 'error', text: 'Failed to load data. Is the API running?' })
    } finally {
      setLoading(false)
    }
  }

  const connectWebSocket = () => {
    ws.current = new WebSocket(WS_URL)
    ws.current.onopen  = () => setWsStatus('connected')
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'history') {
        setAlerts(data.alerts || [])
      }
      if (data.type === 'new_alert') {
        const newAlert = { ...data, _id: data.alert_id }
        setAlerts(prev => [newAlert, ...prev])
        setLiveAlerts(prev => [newAlert, ...prev.slice(0, 9)])
        setTimeout(() => {
          setLiveAlerts(prev =>
            prev.filter(a => a._id !== newAlert._id)
          )
        }, 8000)
      }
    }
    ws.current.onclose = () => {
      setWsStatus('disconnected')
      setTimeout(connectWebSocket, 3000)
    }
    ws.current.onerror = () => setWsStatus('error')
  }

  const updateStatus = async (regNum, status) => {
    try {
      await axios.patch(`${API}/students/${regNum}/status`, { status })
      setStudents(prev =>
        prev.map(s =>
          s.registration_number === regNum ? { ...s, status } : s
        )
      )
      setMsg({ type: 'success', text: `${regNum} updated to ${status}` })
      setTimeout(() => setMsg(null), 3000)
    } catch {
      setMsg({ type: 'error', text: 'Failed to update status' })
    }
  }

  const assignExam = async (regNum, examUrl) => {
    try {
      await axios.patch(`${API}/students/${regNum}/exam`, { exam_url: examUrl })
      setStudents(prev =>
        prev.map(s =>
          s.registration_number === regNum
            ? { ...s, exam_url: examUrl } : s
        )
      )
      setMsg({ type: 'success', text: `Exam URL assigned to ${regNum}` })
      setTimeout(() => setMsg(null), 3000)
    } catch {
      setMsg({ type: 'error', text: 'Failed to assign exam URL' })
    }
  }

  const deleteStudent = async (regNum, note) => {
    await axios.delete(`${API}/students/${regNum}`, { data: { note } })
    setStudents(prev =>
      prev.filter(s => s.registration_number !== regNum)
    )
    setDeleteTarget(null)
    setMsg({
      type: 'success',
      text: `${regNum} was deleted and notified of the reason.`
    })
    setTimeout(() => setMsg(null), 4000)
  }

  const logout = () => {
    if (ws.current) ws.current.close()
    sessionStorage.removeItem('admin')
    navigate('/login')
  }

  const studentAlerts = (regNum) =>
    alerts.filter(a => a.candidate_id === regNum)

  const pending  = students.filter(s => s.status === 'pending').length
  const approved = students.filter(s => s.status === 'approved').length
  const flagged  = students.filter(s => s.status === 'flagged').length

  return (
    <div style={{ padding: '32px', maxWidth: '1200px', margin: '0 auto' }}>

      {/* Modals */}
      {evidenceAlert && (
        <EvidencePopup
          alert={evidenceAlert}
          onClose={() => setEvidenceAlert(null)}
        />
      )}
      {reportStudent && (
        <ReportModal
          student={reportStudent}
          alerts={alerts}
          onClose={() => setReportStudent(null)}
        />
      )}
      {deleteTarget && (
        <DeleteStudentModal
          student={deleteTarget}
          onClose={() => setDeleteTarget(null)}
          onConfirm={deleteStudent}
        />
      )}

      {/* Nav */}
      <div className="nav">
        <div>
          <h2>ExamProctor — Admin Dashboard</h2>
          <p style={{ color: '#64748b', fontSize: '12px', marginTop: '2px' }}>
            Examination Management & Monitoring
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: wsStatus === 'connected'   ? '#22c55e'
                        : wsStatus === 'connecting'  ? '#eab308'
                        : '#ef4444',
              boxShadow: wsStatus === 'connected'
                ? '0 0 6px #22c55e' : 'none'
            }} />
            <span style={{ fontSize: '11px', color: '#64748b' }}>
              {wsStatus === 'connected'    ? 'Live'
               : wsStatus === 'connecting' ? 'Connecting...'
               : 'Disconnected'}
            </span>
          </div>
          <button
            className="btn btn-sm"
            style={{ background: '#334155', color: '#94a3b8', width: 'auto' }}
            onClick={logout}
          >
            Logout
          </button>
        </div>
      </div>

      {/* Live Alert Feed */}
      {liveAlerts.length > 0 && (
        <div style={{ marginBottom: '24px' }}>
          <p style={{
            fontSize: '11px', color: '#64748b',
            textTransform: 'uppercase', letterSpacing: '0.06em',
            marginBottom: '10px', fontWeight: 600
          }}>
            ⚡ Live Alerts
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {liveAlerts.map(a => (
              <div
                key={a._id}
                onClick={() => setEvidenceAlert(a)}
                style={{
                  background:     '#1e293b',
                  border:         `1px solid ${violationColour(a.violation_type)}`,
                  borderLeft:     `4px solid ${violationColour(a.violation_type)}`,
                  borderRadius:   '8px',
                  padding:        '12px 16px',
                  display:        'flex',
                  justifyContent: 'space-between',
                  alignItems:     'center',
                  cursor:         'pointer',
                  transition:     'background 0.2s',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  {a.evidence_frame && (
                    <img
                      src={`data:image/jpeg;base64,${a.evidence_frame}`}
                      alt="Evidence"
                      style={{
                        width: '48px', height: '36px',
                        borderRadius: '4px', objectFit: 'cover',
                        border: `1px solid ${violationColour(a.violation_type)}`
                      }}
                    />
                  )}
                  <div>
                    <span style={{
                      color: violationColour(a.violation_type),
                      fontWeight: 700, fontSize: '13px'
                    }}>
                      ⚠ {a.violation_type}
                    </span>
                    <span style={{
                      color: '#64748b', fontSize: '12px', marginLeft: '12px'
                    }}>
                      {a.candidate_id} · Station {a.station_id}
                    </span>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ color: '#475569', fontSize: '11px' }}>
                    {new Date(a.timestamp).toLocaleTimeString()}
                  </span>
                  <span style={{ color: '#334155', fontSize: '11px' }}>
                    Click to view evidence →
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="stats-row">
        {[
          { number: students.length, label: 'Total Students'   },
          { number: pending,         label: 'Pending Approval' },
          { number: approved,        label: 'Approved'         },
          { number: alerts.length,   label: 'Total Alerts'     },
          { number: flagged,         label: 'Flagged'          },
        ].map(s => (
          <div className="stat-card" key={s.label}>
            <div className="number">{s.number}</div>
            <div className="label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="role-tabs" style={{ maxWidth: '400px' }}>
        {['students', 'reports', 'alerts'].map(v => (
          <button
            key={v}
            className={`role-tab ${view === v ? 'active' : ''}`}
            onClick={() => setView(v)}
          >
            {v.charAt(0).toUpperCase() + v.slice(1)}
            {v === 'alerts' && alerts.length > 0 && (
              <span style={{
                background: '#ef4444', color: 'white',
                borderRadius: '10px', padding: '1px 6px',
                fontSize: '10px', marginLeft: '6px'
              }}>
                {alerts.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {msg && (
        <div className={`msg msg-${msg.type}`} style={{ marginBottom: '16px' }}>
          {msg.text}
        </div>
      )}

      {loading ? (
        <p style={{ color: '#64748b', textAlign: 'center', padding: '40px' }}>
          Loading...
        </p>
      ) : (
        <>
          {/* STUDENTS TAB */}
          {view === 'students' && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Photo</th>
                    <th>Name</th>
                    <th>Reg Number</th>
                    <th>Department</th>
                    <th>Status</th>
                    <th>Exam URL</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {students.length === 0 ? (
                    <tr>
                      <td colSpan="7" style={{
                        textAlign: 'center', color: '#64748b', padding: '40px'
                      }}>
                        No students registered yet
                      </td>
                    </tr>
                  ) : students.map(s => (
                    <tr key={s.registration_number}>
                      <td>
                        {s.passport_photo ? (
                          <img
                            src={`data:${s.passport_mime};base64,${s.passport_photo}`}
                            alt="Passport"
                            style={{
                              width: '44px', height: '44px',
                              borderRadius: '50%', objectFit: 'cover',
                              border: '2px solid #334155'
                            }}
                          />
                        ) : (
                          <div style={{
                            width: '44px', height: '44px',
                            borderRadius: '50%', background: '#334155',
                            display: 'flex', alignItems: 'center',
                            justifyContent: 'center', fontSize: '16px'
                          }}>👤</div>
                        )}
                      </td>
                      <td style={{ fontWeight: 600 }}>{s.full_name}</td>
                      <td style={{ color: '#94a3b8' }}>
                        {s.registration_number}
                      </td>
                      <td>{s.department}</td>
                      <td>
                        <span className={`badge badge-${s.status}`}>
                          {s.status}
                        </span>
                      </td>
                      <td style={{ fontSize: '12px' }}>
                        <div style={{
                          display: 'flex', flexDirection: 'column',
                          gap: '4px', alignItems: 'flex-start'
                        }}>
                          {s.exam_url ? (
                            <a
                              href={s.exam_url}
                              target="_blank"
                              rel="noreferrer"
                              style={{ color: '#38bdf8' }}
                            >
                              View ↗
                            </a>
                          ) : (
                            <span style={{ color: '#475569' }}>
                              Not assigned
                            </span>
                          )}
                          <button
                            className="btn btn-sm"
                            style={{
                              background: '#1e3a5f',
                              color: '#7dd3fc', width: 'auto',
                              padding: '4px 10px', fontSize: '11px'
                            }}
                            onClick={() => {
                              const url = prompt(
                                s.exam_url
                                  ? `Edit exam URL for ${s.full_name}:`
                                  : `Enter exam URL for ${s.full_name}:`,
                                s.exam_url || ''
                              )
                              if (url !== null && url.trim() !== '') {
                                assignExam(s.registration_number, url.trim())
                              }
                            }}
                          >
                            {s.exam_url ? 'Edit URL' : 'Assign URL'}
                          </button>
                        </div>
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          {s.status === 'pending' && (
                            <>
                              <button
                                className="btn btn-success"
                                onClick={() =>
                                  updateStatus(s.registration_number, 'approved')
                                }
                              >
                                Approve
                              </button>
                              <button
                                className="btn btn-danger"
                                onClick={() =>
                                  updateStatus(s.registration_number, 'rejected')
                                }
                              >
                                Reject
                              </button>
                            </>
                          )}
                          {s.status === 'approved' && (
                            <button
                              className="btn btn-danger"
                              onClick={() =>
                                updateStatus(s.registration_number, 'rejected')
                              }
                            >
                              Revoke
                            </button>
                          )}
                          <button
                            className="btn btn-sm"
                            style={{
                              background: '#450a0a',
                              color: '#fca5a5', width: 'auto'
                            }}
                            onClick={() => setDeleteTarget(s)}
                          >
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* REPORTS TAB */}
          {view === 'reports' && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Student</th>
                    <th>Reg Number</th>
                    <th>Violations</th>
                    <th>Integrity Score</th>
                    <th>Grade</th>
                    <th>Breakdown</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {students.filter(s =>
                    studentAlerts(s.registration_number).length > 0
                  ).length === 0 ? (
                    <tr>
                      <td colSpan="7" style={{
                        textAlign: 'center', color: '#64748b', padding: '40px'
                      }}>
                        No exam reports available yet
                      </td>
                    </tr>
                  ) : students.map(s => {
                    const sAlerts = studentAlerts(s.registration_number)
                    if (sAlerts.length === 0) return null
                    const score = calcScore(sAlerts)
                    const types = {}
                    sAlerts.forEach(a => {
                      const t = a.violation_type || 'Unknown'
                      types[t] = (types[t] || 0) + 1
                    })
                    return (
                      <tr key={s.registration_number}>
                        <td style={{ fontWeight: 600 }}>{s.full_name}</td>
                        <td style={{ color: '#94a3b8' }}>
                          {s.registration_number}
                        </td>
                        <td style={{ textAlign: 'center', fontWeight: 700 }}>
                          {sAlerts.length}
                        </td>
                        <td>
                          <span
                            className={scoreClass(score)}
                            style={{ fontSize: '18px' }}
                          >
                            {score}%
                          </span>
                        </td>
                        <td>
                          <span
                            className={scoreClass(score)}
                            style={{ fontSize: '12px' }}
                          >
                            {scoreLabel(score)}
                          </span>
                        </td>
                        <td style={{ fontSize: '11px', color: '#94a3b8' }}>
                          {Object.entries(types).map(([t, c]) => (
                            <div key={t} style={{ marginBottom: '2px' }}>
                              {t}: <strong>{c}x</strong>
                            </div>
                          ))}
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: '6px' }}>
                            <button
                              className="btn btn-sm"
                              style={{
                                background: '#1e3a5f',
                                color: '#7dd3fc', width: 'auto'
                              }}
                              onClick={() => setReportStudent(s)}
                            >
                              View Report
                            </button>
                            <a
                              href={`${API}/reports/${s.registration_number}`}
                              target="_blank"
                              rel="noreferrer"
                              style={{
                                padding:        '7px 14px',
                                background:     '#064e3b',
                                color:          '#6ee7b7',
                                borderRadius:   '6px',
                                fontSize:       '12px',
                                fontWeight:     700,
                                textDecoration: 'none',
                              }}
                            >
                              ↓ PDF
                            </a>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* ALERTS TAB */}
          {view === 'alerts' && (
            <div className="table-wrap">
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', marginBottom: '16px'
              }}>
                <p style={{ color: '#64748b', fontSize: '13px' }}>
                  {alerts.length} total alerts recorded
                </p>
                <button
                  className="btn btn-sm"
                  style={{
                    background: '#450a0a', color: '#fca5a5', width: 'auto'
                  }}
                  onClick={async () => {
                    if (!window.confirm('Clear all alerts?')) return
                    await axios.delete(`${API}/alerts`)
                    setAlerts([])
                    setLiveAlerts([])
                  }}
                >
                  Clear All Alerts
                </button>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Evidence</th>
                    <th>Time</th>
                    <th>Candidate</th>
                    <th>Station</th>
                    <th>Violation</th>
                    <th>Objects</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.length === 0 ? (
                    <tr>
                      <td colSpan="6" style={{
                        textAlign: 'center', color: '#64748b', padding: '40px'
                      }}>
                        No alerts recorded yet
                      </td>
                    </tr>
                  ) : alerts.map(a => (
                    <tr
                      key={a._id}
                      style={{ cursor: 'pointer' }}
                      onClick={() => setEvidenceAlert(a)}
                    >
                      <td>
                        {a.evidence_frame ? (
                          <img
                            src={`data:image/jpeg;base64,${a.evidence_frame}`}
                            alt="Evidence"
                            style={{
                              width: '64px', height: '48px',
                              borderRadius: '6px', objectFit: 'cover',
                              border: `1px solid ${violationColour(a.violation_type)}`
                            }}
                          />
                        ) : (
                          <div style={{
                            width: '64px', height: '48px',
                            borderRadius: '6px', background: '#1e293b',
                            display: 'flex', alignItems: 'center',
                            justifyContent: 'center',
                            color: '#334155', fontSize: '10px'
                          }}>
                            No frame
                          </div>
                        )}
                      </td>
                      <td style={{ color: '#64748b', fontSize: '12px' }}>
                        {new Date(a.timestamp).toLocaleString()}
                      </td>
                      <td style={{ fontWeight: 600 }}>{a.candidate_id}</td>
                      <td style={{ color: '#94a3b8' }}>{a.station_id}</td>
                      <td>
                        <span style={{
                          color:      violationColour(a.violation_type),
                          fontWeight: 600, fontSize: '12px'
                        }}>
                          {a.violation_type}
                        </span>
                      </td>
                      <td style={{ color: '#94a3b8', fontSize: '12px' }}>
                        {a.objects?.join(', ') || '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}