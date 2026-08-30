export type TankId = string;
export type Role = 'user' | 'admin';

export interface HistoryPoint {
  item_id: string;
  timestamp: string;
  'Outflow in KL': number;
}

export interface ForecastPoint {
  item_id: string;
  timestamp: string;
  pred_mean: number;
  pred_p10?: number;
  pred_p50?: number;
  pred_p90?: number;
}

export interface ForecastResponse {
  tank_id: string;
  prediction_length: number;
  history: HistoryPoint[];
  forecasts: Record<string, ForecastPoint[]>;
  forecast_sources: Record<string, string>;
  warnings: string[];
}

export interface HistoryResponse {
  tank_id: string;
  hours: number;
  history: HistoryPoint[];
}

export interface TanksResponse {
  tanks: TankId[];
}

export interface TaskEnqueued {
  status: 'queued';
  task_id: string;
  task_name: string;
}

export interface AuthedUser {
  sub: string;
  role: Role;
  locationId?: string;
}
