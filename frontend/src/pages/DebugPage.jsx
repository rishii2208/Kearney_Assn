import styles from './Page.module.css'

function DebugPage() {
  return (
    <section className={styles.page}>
      <h2 className={styles.heading}>Debug</h2>
      <p className={styles.text}>Use this page to inspect request and index behavior.</p>
    </section>
  )
}

export default DebugPage
