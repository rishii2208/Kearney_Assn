import styles from './Page.module.css'

function EvalPage() {
  return (
    <section className={styles.page}>
      <h2 className={styles.heading}>Eval</h2>
      <p className={styles.text}>Inspect offline evaluation runs and ranking quality.</p>
    </section>
  )
}

export default EvalPage
