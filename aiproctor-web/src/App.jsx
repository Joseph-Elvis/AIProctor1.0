import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Login from './pages/Login'
import StudentRegister from './pages/StudentRegister'
import StudentDashboard from './pages/StudentDashboard'
import AdminDashboard from './pages/AdminDashboard'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"                    element={<Navigate to="/login" />} />
        <Route path="/login"               element={<Login />} />
        <Route path="/student/register"    element={<StudentRegister />} />
        <Route path="/student/dashboard"   element={<StudentDashboard />} />
        <Route path="/admin/dashboard"     element={<AdminDashboard />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App