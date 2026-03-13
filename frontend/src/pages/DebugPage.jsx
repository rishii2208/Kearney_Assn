import { useEffect, useState } from 'react'
import pageStyles from './Page.module.css'
import styles from './DebugPage.module.css'

function DebugPage() {
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [severity, setSeverity] = useState('all')
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchLogs = async (event) => {
    if (event) {
      event.preventDefault()
    }

    setLoading(true)
    setError('')

    try {
      const params = new URLSearchParams({
        limit: '200',
        severity,
      })

      if (startDate) {
        params.set('start_date', startDate)
      }
      if (endDate) {
        params.set('end_date', endDate)
      }

      const response = await fetch(`/logs?${params.toString()}`)
      const payload = await response.json().catch(() => ({}))

      if (!response.ok) {
        throw new Error(payload?.detail || 'Failed to fetch logs.')
      }

      setLogs(Array.isArray(payload.logs) ? payload.logs : [])
    } catch (requestError) {
      setLogs([])
      setError(requestError.message || 'Failed to fetch logs.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
  }, [])

  return (
    <section className={pageStyles.page}>
      <h2 className={pageStyles.heading}>Debug</h2>

      <form className={styles.filters} onSubmit={fetchLogs}>
        <label className={styles.field}>
          <span className={styles.label}>Start date</span>
          <input
            className={styles.input}
            type="date"
            value={startDate}
            onChange={(event) => setStartDate(event.target.value)}
          />
        </label>

        <label className={styles.field}>
          <span className={styles.label}>End date</span>
          <input
            className={styles.input}
            type="date"
            value={endDate}
            onChange={(event) => setEndDate(event.target.value)}
          />
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Severity</span>
          <select
            className={styles.input}
            value={severity}
            onChange={(event) => setSeverity(event.target.value)}
          >
            <option value="all">All</option>
            <option value="error">Error only</option>
            <option value="success">Success only</option>
          </select>
        </label>

        <button className={styles.button} type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Apply'}
        </button>
      </form>

      {error && <p className={styles.error}>Error: {error}</p>}

      {logs.length === 0 && !loading && !error ? (
        <p className={styles.empty}>No logs found for the selected filters.</p>
      ) : (
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Created</th>
                <th>Request ID</th>
                <th>Query</th>
                <th>Latency (ms)</th>
                <th>Top K</th>
                <th>Alpha</th>
                <th>Result Count</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((row) => {
                const hasError = Boolean(row.error && String(row.error).trim())
                return (
                  <tr key={row.request_id} className={hasError ? styles.errorRow : ''}>
                    <td>{row.created_at || '-'}</td>
                    <td>{row.request_id || '-'}</td>
                    <td>{row.query || '-'}</td>
                    <td>{Number(row.latency_ms || 0).toFixed(2)}</td>
                    <td>{row.top_k ?? '-'}</td>
                    <td>{Number(row.alpha || 0).toFixed(2)}</td>
                    <td>{row.result_count ?? '-'}</td>
                    <td>{row.error || '-'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default DebugPage
