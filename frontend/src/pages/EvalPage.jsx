import { useEffect, useMemo, useState } from 'react'
import pageStyles from './Page.module.css'
import styles from './EvalPage.module.css'

function NdcgChart({ points }) {
  if (points.length === 0) {
    return <p className={styles.empty}>No ndcg@10 data available.</p>
  }

  const width = 760
  const height = 260
  const padding = 32
  const usableWidth = width - padding * 2
  const usableHeight = height - padding * 2
  const maxY = Math.max(1, ...points.map((point) => point.ndcg))
  const minY = 0

  const chartPoints = points.map((point, index) => {
    const x =
      points.length === 1
        ? width / 2
        : padding + (index / (points.length - 1)) * usableWidth
    const yRatio = (point.ndcg - minY) / (maxY - minY || 1)
    const y = height - padding - yRatio * usableHeight
    return { ...point, x, y }
  })

  const linePath = chartPoints
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ')

  return (
    <div className={styles.chartWrap}>
      <svg className={styles.chart} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="ndcg at 10 chart">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className={styles.axis} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className={styles.axis} />
        <path d={linePath} className={styles.line} />
        {chartPoints.map((point) => (
          <circle key={point.run} cx={point.x} cy={point.y} r="3.5" className={styles.dot}>
            <title>{`Run ${point.run}: ${point.ndcg.toFixed(4)}`}</title>
          </circle>
        ))}
        <text x={padding - 8} y={padding + 4} className={styles.axisLabel} textAnchor="end">
          {maxY.toFixed(2)}
        </text>
        <text x={padding - 8} y={height - padding + 4} className={styles.axisLabel} textAnchor="end">
          0.00
        </text>
      </svg>
    </div>
  )
}

function EvalPage() {
  const [experiments, setExperiments] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const fetchExperiments = async () => {
    setLoading(true)
    setError('')

    try {
      const response = await fetch('/experiments')
      if (!response.ok) {
        throw new Error('Failed to fetch experiments')
      }

      const payload = await response.json()
      const rows = Array.isArray(payload.experiments) ? payload.experiments : []
      setExperiments(rows)
    } catch (requestError) {
      setExperiments([])
      setError(requestError.message || 'Failed to load experiments.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchExperiments()
  }, [])

  const chartPoints = useMemo(
    () =>
      experiments
        .map((row, index) => ({
          run: index + 1,
          ndcg: Number(row.ndcg_at_10),
        }))
        .filter((row) => Number.isFinite(row.ndcg)),
    [experiments],
  )

  return (
    <section className={pageStyles.page}>
      <div className={styles.headerRow}>
        <h2 className={pageStyles.heading}>Eval</h2>
        <button className={styles.refreshButton} onClick={fetchExperiments} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && <p className={styles.error}>Error: {error}</p>}

      <section className={styles.panel}>
        <h3 className={styles.panelTitle}>ndcg@10 over runs</h3>
        <NdcgChart points={chartPoints} />
      </section>

      <section className={styles.panel}>
        <h3 className={styles.panelTitle}>Experiment Runs</h3>
        {experiments.length === 0 ? (
          <p className={styles.empty}>No experiment rows found.</p>
        ) : (
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Timestamp</th>
                  <th>Commit</th>
                  <th>ndcg@10</th>
                  <th>recall@10</th>
                  <th>mrr@10</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((row, index) => (
                  <tr key={`${row.timestamp || 'run'}-${index}`}>
                    <td>{index + 1}</td>
                    <td>{row.timestamp || '-'}</td>
                    <td>{row.git_commit || '-'}</td>
                    <td>{Number(row.ndcg_at_10).toFixed(4)}</td>
                    <td>{Number(row.recall_at_10).toFixed(4)}</td>
                    <td>{Number(row.mrr_at_10).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  )
}

export default EvalPage
