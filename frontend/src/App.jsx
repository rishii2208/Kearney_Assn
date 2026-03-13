import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import styles from './App.module.css'
import SearchPage from './pages/SearchPage.jsx'
import KpisPage from './pages/KpisPage.jsx'
import EvalPage from './pages/EvalPage.jsx'
import DebugPage from './pages/DebugPage.jsx'

const links = [
  { to: '/', label: 'Search' },
  { to: '/kpis', label: 'KPIs' },
  { to: '/eval', label: 'Eval' },
  { to: '/debug', label: 'Debug' },
]

function App() {
  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <h1 className={styles.title}>Search Dashboard</h1>
        <nav className={styles.nav}>
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) =>
                isActive ? `${styles.navLink} ${styles.active}` : styles.navLink
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className={styles.main}>
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/kpis" element={<KpisPage />} />
          <Route path="/eval" element={<EvalPage />} />
          <Route path="/debug" element={<DebugPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
