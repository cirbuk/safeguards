import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface MetricsData {
  timestamp: string;
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  network_mbps: number;
}

export interface BudgetData {
  timestamp: string;
  usage_percent: number;
  allocation: number;
  used: number;
}

export interface AgentMetrics {
  agent_id: string;
  name: string;
  status: string;
  metrics: MetricsData;
  budget: BudgetData;
}

export interface TrendAnalysis {
  trend_direction: string;
  rate_of_change: number;
  volatility: number;
  forecast_next_hour: number;
  forecast_next_day: number;
  anomaly_score: number;
}

export interface UsagePattern {
  peak_hours: number[];
  low_usage_hours: number[];
  weekly_pattern: Record<string, number>;
  periodic_spikes: string[];
  correlation_matrix: Record<string, Record<string, number>>;
}

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const metricsApi = {
  // Real-time metrics
  getRealtimeMetrics: async (agentId: string): Promise<AgentMetrics> => {
    const response = await api.get(`/api/metrics/realtime/${agentId}`);
    return response.data;
  },

  // Historical metrics
  getHistoricalMetrics: async (
    agentId: string,
    startTime: string,
    endTime: string
  ): Promise<MetricsData[]> => {
    const response = await api.get(`/api/metrics/historical/${agentId}`, {
      params: { start_time: startTime, end_time: endTime },
    });
    return response.data;
  },

  // Trend analysis
  getTrendAnalysis: async (
    agentId: string,
    metricName: string
  ): Promise<TrendAnalysis> => {
    const response = await api.get(`/api/metrics/trends/${agentId}`, {
      params: { metric_name: metricName },
    });
    return response.data;
  },

  // Usage patterns
  getUsagePatterns: async (agentId: string): Promise<UsagePattern> => {
    const response = await api.get(`/api/metrics/patterns/${agentId}`);
    return response.data;
  },

  // Budget metrics
  getBudgetMetrics: async (agentId: string): Promise<BudgetData[]> => {
    const response = await api.get(`/api/budget/${agentId}`);
    return response.data;
  },

  // System overview
  getSystemOverview: async (): Promise<{
    total_agents: number;
    active_agents: number;
    total_budget_usage: number;
    system_metrics: MetricsData;
  }> => {
    const response = await api.get('/api/metrics/overview');
    return response.data;
  },
};

// Add authentication interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
