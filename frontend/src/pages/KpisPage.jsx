import styles from './Page.module.css'

function KpisPage() {
  return (
    <section className={styles.page}>
      <h2 className={styles.heading}>KPIs</h2>
      <p className={styles.text}>View request volume, errors, and latency metrics.</p>
    </section>
  )
}

export default KpisPage
