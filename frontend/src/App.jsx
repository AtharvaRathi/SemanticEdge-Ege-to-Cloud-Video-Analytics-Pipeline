import React, { useState, useEffect } from 'react';
import {
  Search,
  Video,
  Clock,
  Activity,
  Layers,
  Cpu,
  Shield,
  SlidersHorizontal,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  Sparkles,
  ChevronDown
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const EXAMPLE_QUERIES = [
  'person in black hoodie running',
  'white delivery van stopping',
  'bicycle crossing the street',
  'crowd gathering near entrance',
  'unattended package or box',
];

export default function App() {
  const [query, setQuery] = useState('');
  const [userId, setUserId] = useState('edge_default');
  const [limit, setLimit] = useState(10);
  const [minScore, setMinScore] = useState(0.0);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searched, setSearched] = useState(false);
  const [searchTimeMs, setSearchTimeMs] = useState(null);
  const [apiHealth, setApiHealth] = useState({ status: 'checking', message: '' });

  // Probe API health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/health`, {
        headers: { 'Accept': 'application/json' },
      });
      if (res.ok) {
        const data = await res.json();
        const qdrantOk = data?.qdrant?.qdrant === 'ok' || data?.qdrant?.status === 'ok';
        setApiHealth({
          status: qdrantOk ? 'online' : 'degraded',
          message: qdrantOk ? 'Backend & Qdrant Connected' : 'Qdrant offline / connecting',
        });
      } else {
        setApiHealth({ status: 'offline', message: `HTTP ${res.status}` });
      }
    } catch (err) {
      setApiHealth({ status: 'offline', message: 'Backend unreachable' });
    }
  };

  const handleSearch = async (e, forcedQuery = null) => {
    if (e && e.preventDefault) e.preventDefault();
    const searchQuery = forcedQuery !== null ? forcedQuery : query;
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    setSearched(true);
    const startTime = performance.now();

    try {
      // Connect to FastAPI /search endpoint
      const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery.trim(),
          user_id: userId.trim() || 'edge_default',
          limit: Number(limit) || 10,
          min_score: Number(minScore) !== undefined ? Number(minScore) : 0.15,
        }),
      });

      const elapsed = Math.round(performance.now() - startTime);
      setSearchTimeMs(elapsed);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Search failed with status ${response.status}`);
      }

      const data = await response.json();
      setResults(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Search request error:', err);
      setError(err.message || 'Failed to communicate with cloud search engine.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (ts) => {
    if (!ts) return 'N/A';
    // If epoch float in seconds
    const date = new Date(ts > 1e11 ? ts : ts * 1000);
    return isNaN(date.getTime()) ? String(ts) : date.toLocaleTimeString() + ` (${date.toLocaleDateString()})`;
  };

  const getScoreStyle = (score) => {
    if (score >= 0.75) return { className: 'score-high', label: 'High Match' };
    if (score >= 0.50) return { className: 'score-med', label: 'Moderate' };
    return { className: 'score-low', label: 'Weak' };
  };

  return (
    <div className="container">
      {/* ── Header ────────────────────────────────────────────── */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={22} color="#ffffff" />
          </div>
          <div>
            <h1 className="brand-title">SemanticEdge</h1>
            <p className="brand-subtitle">Edge-to-Cloud Zero-Shot Video Search</p>
          </div>
        </div>

        <div className="status-badge">
          <span className={`status-dot ${apiHealth.status === 'offline' ? 'error' : ''}`} />
          <span>{apiHealth.message || 'System Initializing'}</span>
          <button
            onClick={checkHealth}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', display: 'flex' }}
            title="Refresh status"
          >
            <RefreshCw size={12} />
          </button>
        </div>
      </header>

      {/* ── Error Banner ───────────────────────────────────────── */}
      {error && (
        <div className="error-banner">
          <AlertCircle size={18} />
          <div>
            <strong>Error:</strong> {error}
            {error.includes('unreachable') && (
              <span style={{ display: 'block', fontSize: '0.8rem', marginTop: '0.25rem', opacity: 0.85 }}>
                Ensure the FastAPI backend is running on <code>{API_BASE_URL}</code> or verify <code>docker-compose up</code>.
              </span>
            )}
          </div>
        </div>
      )}

      {/* ── Search Input Card ─────────────────────────────────── */}
      <div className="search-card">
        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-wrapper">
            <Search className="search-icon-left" size={20} />
            <input
              type="text"
              className="search-input"
              placeholder="Search video clips using natural language (e.g. 'person carrying black backpack', 'red sedan stopping')..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              autoFocus
            />
            <button
              type="submit"
              className="search-submit-btn"
              disabled={loading || !query.trim()}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Search</span>
                </>
              )}
            </button>
          </div>

          {/* Quick presets */}
          <div className="controls-row">
            <div className="presets-group">
              <span className="presets-label">Try query:</span>
              {EXAMPLE_QUERIES.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  className="preset-chip"
                  onClick={() => {
                    setQuery(preset);
                    handleSearch(null, preset);
                  }}
                >
                  {preset}
                </button>
              ))}
            </div>

            {/* Config controls */}
            <div className="options-group">
              <div className="control-item">
                <SlidersHorizontal size={14} />
                <span>Limit:</span>
                <select
                  className="control-select"
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                >
                  <option value={5}>Top 5</option>
                  <option value={10}>Top 10</option>
                  <option value={20}>Top 20</option>
                  <option value={50}>Top 50</option>
                </select>
              </div>

              <div className="control-item">
                <span>Threshold:</span>
                <select
                  className="control-select"
                  value={minScore}
                  onChange={(e) => setMinScore(Number(e.target.value))}
                >
                  <option value={0.00}>&ge; 0% (All Positive)</option>
                  <option value={0.05}>&ge; 5% (Moderate)</option>
                  <option value={0.10}>&ge; 10% (High)</option>
                  <option value={0.15}>&ge; 15% (Strict)</option>
                </select>
              </div>

              <div className="control-item">
                <span>User/Stream ID:</span>
                <input
                  type="text"
                  className="control-input"
                  style={{ width: '110px' }}
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="edge_default"
                />
              </div>
            </div>
          </div>
        </form>
      </div>

      {/* ── Results Area ───────────────────────────────────────── */}
      {searched && (
        <div style={{ marginBottom: '2rem' }}>
          <div className="results-header">
            <div className="results-title">
              <Layers size={18} color="#3b82f6" />
              <span>Matching Video Keyframes</span>
              <span className="pill pill-highlight" style={{ marginLeft: '0.5rem' }}>
                {results.length} found
              </span>
            </div>
            {searchTimeMs !== null && (
              <span className="results-meta">
                Latency: {searchTimeMs}ms (SigLIP ONNX + Qdrant HNSW)
              </span>
            )}
          </div>

          {results.length === 0 && !loading && (
            <div className="state-box">
              <Video className="state-icon" size={48} />
              <h3 className="state-title">No matching keyframes found</h3>
              <p className="state-desc">
                No video events matched "{query}" for tenant <code>{userId}</code> with score &ge; {Math.round(minScore * 100)}%.
                Make sure video keyframes have been ingested via <code>/edge/edge_ingest.py</code> or uploaded to the cloud service.
              </p>
            </div>
          )}

          <div className="results-grid">
            {results.map((item, idx) => {
              const clampedScore = Math.max(0, Math.min(1.0, item.score ?? 0));
              const matchPercentage = (clampedScore * 100).toFixed(1);
              const scoreStyle = getScoreStyle(clampedScore);
              const metadata = item.metadata || {};
              const objectCount = metadata.object_count ?? null;
              const motionScore = metadata.motion_score ?? null;
              const trackerIds = metadata.tracker_ids ?? [];

              return (
                <div key={idx} className="result-card">
                  {item.image && (
                    <div className="card-image-wrapper">
                      <img
                        className="card-image"
                        src={item.image}
                        alt={`Keyframe #${item.frame_idx !== undefined ? item.frame_idx : idx} from ${item.video_id || item.filename || 'stream'}`}
                        loading="lazy"
                        onError={(e) => {
                          // Fallback to placeholder if image load fails
                          e.target.style.display = 'none';
                        }}
                      />
                      <div className="card-image-overlay">
                        <span className="frame-overlay-tag">
                          Frame #{item.frame_idx !== undefined ? item.frame_idx : idx}
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="card-top">
                    <div className="video-badge">
                      <Video size={14} />
                      <span>{item.filename || item.video_id || `Stream #${idx + 1}`}</span>
                    </div>
                    <div className={`score-badge ${scoreStyle.className}`} title={`Cosine Similarity: ${item.score}`}>
                      {matchPercentage}% match
                    </div>
                  </div>

                  <div className="card-body">
                    <div className="metrics-row">
                      <div className="metric-item">
                        <span className="metric-label">Timestamp</span>
                        <span className="metric-value" style={{ fontSize: '0.78rem' }}>
                          {formatTimestamp(item.timestamp)}
                        </span>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">Frame Index</span>
                        <span className="metric-value">
                          #{item.frame_idx !== undefined ? item.frame_idx : 'N/A'}
                        </span>
                      </div>
                    </div>

                    {/* Metadata tags */}
                    <div className="metadata-pills">
                      {objectCount !== null && (
                        <span className="pill pill-highlight">
                          {objectCount} {objectCount === 1 ? 'object' : 'objects'} tracked
                        </span>
                      )}
                      {motionScore !== null && (
                        <span className="pill">
                          motion: {(motionScore * 100).toFixed(1)}%
                        </span>
                      )}
                      {trackerIds.length > 0 && (
                        <span className="pill">
                          tracks: #{trackerIds.slice(0, 4).join(', #')}
                          {trackerIds.length > 4 ? '...' : ''}
                        </span>
                      )}
                    </div>

                    {/* Deep inspection for raw metadata */}
                    {Object.keys(metadata).length > 0 && (
                      <details className="raw-json-details">
                        <summary className="raw-json-summary">
                          View Edge Metadata ({Object.keys(metadata).length} fields)
                        </summary>
                        <pre className="raw-json-content">
                          {JSON.stringify(metadata, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Default Initial State ───────────────────────────────── */}
      {!searched && (
        <div className="state-box">
          <Activity className="state-icon" size={42} color="#3b82f6" />
          <h3 className="state-title">Ready for Natural Language Video Queries</h3>
          <p className="state-desc">
            Type any natural scene description above. SemanticEdge matches your text prompt against
            768-dimensional SigLIP vision embeddings indexed in Qdrant with sub-10ms latency.
          </p>
        </div>
      )}
    </div>
  );
}
