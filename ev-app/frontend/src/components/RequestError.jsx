export function RequestError({ error }) {
  return error ? <p role="alert" className="request-error">{error} Refresh to retry.</p> : null
}
