import { useState } from 'react'
import pageStyles from './Page.module.css'
import styles from './SearchPage.module.css'

function SearchPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(10)
  const [alpha, setAlpha] = useState(0.5)
  const [results, setResults] = useState([])
  const [latencyMs, setLatencyMs] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()

    const trimmedQuery = query.trim()
    if (!trimmedQuery) {
      setError('Query is required.')
      setResults([])
      setHasSearched(false)
      return
    }

    setLoading(true)
    setError('')
    setHasSearched(false)

    const started = performance.now()

    try {
      const response = await fetch('/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: trimmedQuery,
          top_k: Number(topK),
          alpha: Number(alpha),
        }),
      })

      const payload = await response.json().catch(() => ({}))
      if (!response.ok) {
        const message = payload?.detail || 'Search request failed.'
        throw new Error(message)
      }

      const rows = Array.isArray(payload.results) ? payload.results : []
      setResults(rows)
      setHasSearched(true)
    } catch (requestError) {
      setResults([])
      setHasSearched(true)
      setError(requestError.message || 'Failed to search. Please try again.')
    } finally {
      setLatencyMs(performance.now() - started)
      setLoading(false)
    }
  }

  return (
    <section className={pageStyles.page}>
      <h2 className={pageStyles.heading}>Search</h2>

      <form className={styles.form} onSubmit={handleSubmit}>
        <label className={styles.field}>
          <span className={styles.label}>Query</span>
          <input
            className={styles.input}
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            maxLength={500}
            placeholder="Search the corpus..."
          />
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Top K</span>
          <input
            className={styles.input}
            type="number"
            min={1}
            max={100}
            value={topK}
            onChange={(event) => setTopK(event.target.value)}
          />
        </label>

        <label className={styles.field}>
          <span className={styles.label}>Alpha: {Number(alpha).toFixed(2)}</span>
          <input
            className={styles.slider}
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={alpha}
            onChange={(event) => setAlpha(event.target.value)}
          />
        </label>

        <button className={styles.button} type="submit" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {latencyMs !== null && (
        <p className={styles.meta}>Latency: {latencyMs.toFixed(1)} ms</p>
      )}

      {error && <p className={styles.error}>Error: {error}</p>}

      {!error && hasSearched && results.length === 0 && (
        <p className={styles.empty}>No results found.</p>
      )}

      {results.length > 0 && (
        <ul className={styles.results}>
          {results.map((result) => (
            <li key={result.doc_id} className={styles.card}>
              <h3 className={styles.resultTitle}>{result.title || result.doc_id}</h3>
              <p className={styles.snippet}>{result.snippet || 'No snippet available.'}</p>
              <p className={styles.scores}>
                BM25: {Number(result.bm25_score).toFixed(4)} | Vector:{' '}
                {Number(result.vector_score).toFixed(4)} | Hybrid:{' '}
                {Number(result.hybrid_score).toFixed(4)}
              </p>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default SearchPage
