import { useEffect, useState } from 'react'
import pageStyles from './Page.module.css'
import styles from './KpisPage.module.css'

function parsePrometheusText(metricsText) {
  const values = {}
  const lines = metricsText.split('\n')

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) {
      continue
    }

    const parts = trimmed.split(/\s+/)
    if (parts.length < 2) {
      continue
    }

    const metricName = parts[0]
    const metricValue = Number(parts[parts.length - 1])
    if (!Number.isNaN(metricValue)) {
      values[metricName] = metricValue
    }
  }

  return {
    requestCount: values.search_requests_total ?? 0,
    errorCount: values.search_errors_total ?? 0,
    p50LatencyMs: values.search_latency_p50_ms ?? 0,
    p95LatencyMs: values.search_latency_p95_ms ?? 0,
    zeroResultCount: values.search_zero_result_queries_total ?? 0,
  }
}

function KpisPage() {
  const [metrics, setMetrics] = useState({
    requestCount: 0,
    errorCount: 0,
    p50LatencyMs: 0,
    p95LatencyMs: 0,
    zeroResultCount: 0,
  })
  const [topQueries, setTopQueries] = useState([])
  const [zeroResultQueries, setZeroResultQueries] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  const loadKpis = async () => {
    setLoading(true)
    setError('')

    try {
      const [metricsResponse, topQueriesResponse, zeroResultResponse] = await Promise.all([
        fetch('/metrics'),
        fetch('/metrics/top-queries?limit=5'),
        fetch('/metrics/zero-result-queries?limit=5'),
      ])

      if (!metricsResponse.ok) {
        throw new Error('Failed to fetch /metrics')
      }
      if (!topQueriesResponse.ok) {
        throw new Error('Failed to fetch top queries')
      }
      if (!zeroResultResponse.ok) {
        throw new Error('Failed to fetch zero-result queries')
      }

      const [metricsText, topQueriesPayload, zeroResultPayload] = await Promise.all([
        metricsResponse.text(),
        topQueriesResponse.json(),
        zeroResultResponse.json(),
      ])

      setMetrics(parsePrometheusText(metricsText))
      setTopQueries(Array.isArray(topQueriesPayload.top_queries) ? topQueriesPayload.top_queries : [])
      setZeroResultQueries(
        Array.isArray(zeroResultPayload.zero_result_queries)
          ? zeroResultPayload.zero_result_queries
          : [],
      )
      setLastUpdated(new Date())
    } catch (requestError) {
      setError(requestError.message || 'Failed to load KPI data.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadKpis()
  }, [])

  return (
    <section className={pageStyles.page}>
      <div className={styles.headerRow}>
        <h2 className={pageStyles.heading}>KPIs</h2>
        <button className={styles.refreshButton} type="button" onClick={loadKpis} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <p className={styles.updatedAt}>
        Last updated: {lastUpdated ? lastUpdated.toLocaleString() : 'Not loaded yet'}
      </p>

      {error && <p className={styles.error}>Error: {error}</p>}

      <div className={styles.grid}>
        <article className={styles.card}>
          <h3 className={styles.cardTitle}>Requests</h3>
          <p className={styles.value}>{metrics.requestCount}</p>
        </article>
        <article className={styles.card}>
          <h3 className={styles.cardTitle}>Errors</h3>
          <p className={styles.value}>{metrics.errorCount}</p>
        </article>
        <article className={styles.card}>
          <h3 className={styles.cardTitle}>p50 Latency</h3>
          <p className={styles.value}>{metrics.p50LatencyMs.toFixed(2)} ms</p>
        </article>
        <article className={styles.card}>
          <h3 className={styles.cardTitle}>p95 Latency</h3>
          <p className={styles.value}>{metrics.p95LatencyMs.toFixed(2)} ms</p>
        </article>
        <article className={styles.card}>
          <h3 className={styles.cardTitle}>Zero-result count</h3>
          <p className={styles.value}>{metrics.zeroResultCount}</p>
        </article>
      </div>

      <div className={styles.tables}>
        <section className={styles.listCard}>
          <h3 className={styles.cardTitle}>Top Queries</h3>
          {topQueries.length === 0 ? (
            <p className={styles.empty}>No query data yet.</p>
          ) : (
            <ul className={styles.list}>
              {topQueries.map((row) => (
                <li key={`top-${row.query}`} className={styles.listItem}>
                  <span className={styles.queryText}>{row.query}</span>
                  <span className={styles.count}>{row.count}</span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className={styles.listCard}>
          <h3 className={styles.cardTitle}>Zero-result Queries</h3>
          {zeroResultQueries.length === 0 ? (
            <p className={styles.empty}>No zero-result queries yet.</p>
          ) : (
            <ul className={styles.list}>
              {zeroResultQueries.map((row) => (
                <li key={`zero-${row.query}`} className={styles.listItem}>
                  <span className={styles.queryText}>{row.query}</span>
                  <span className={styles.count}>{row.count}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </section>
  )
}

export default KpisPage
