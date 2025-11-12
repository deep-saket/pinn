import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';

export interface TrainingHistoryEntry {
  epoch: number;
  train_data_loss: number;
  train_pde_loss: number;
  train_total_loss: number;
  val_data_loss?: number | null;
  val_pde_loss?: number | null;
  val_total_loss?: number | null;
}

export interface SimulateResponse {
  message: string;
  dataset_size: number;
  target_time: number;
  x_grid: number[];
  target_profile: number[];
  training_history?: TrainingHistoryEntry[];
  last_trained_at?: string;
  last_trained_epochs?: number;
}

export interface IterationPayload {
  iteration: number;
  D: number;
  v: number;
  cost: number;
  u_profile: number[];
  du_dx_profile?: number[];
  grad_D?: number;
  grad_V?: number;
  grad_norm?: number;
  outlet_pressure?: number;
  mean_pressure?: number;
  outlet_gradient?: number;
}

export interface OptimizationResult {
  best_params: { D: number; v: number };
  best_cost: number;
  best_profile: number[];
  best_gradient_profile?: number[];
  best_metrics?: {
    outlet_pressure?: number;
    mean_pressure?: number;
    outlet_gradient?: number;
  };
}

export interface OptimizationHistoryResponse {
  history: IterationPayload[];
  result: OptimizationResult | null;
}

export interface SurrogateStatus {
  has_model: boolean;
  model_path?: string;
  dataset_size?: number;
  last_trained_epochs?: number;
  last_trained_at?: string;
  training_history?: TrainingHistoryEntry[];
  sample_prediction?: {
    x: number[];
    u_hat: number[];
    t: number;
    D: number;
    v: number;
    du_dx?: number[];
  };
}

export interface FacilityStatus {
  has_model: boolean;
  model_path?: string;
  dataset_size?: number;
  last_trained_at?: string;
  training_history?: { epoch: number; train_loss: number; val_loss?: number | null }[];
}

export interface PipelineRunResponse {
  message: string;
  pipe: {
    x: number[];
    profile: number[];
    outlet_pressure: number;
    mean_pressure: number;
    du_dx: number[];
  };
  facility: {
    vapor_fraction: number;
    gas_flow_rate: number;
    liquid_flow_rate: number;
  };
  facility_features: number[];
}

export interface FacilityOptimizeHistoryEntry {
  iteration: number;
  separator_pressure: number;
  separator_temp: number;
  gas_fraction: number;
  cost: number;
  metrics: {
    vapor_fraction?: number;
    gas_flow_rate?: number;
    liquid_flow_rate?: number;
  };
  gradients: {
    dJ_dP?: number;
    dJ_dT?: number;
    dJ_dGas?: number;
  };
}

export interface FacilityOptimizeResult {
  best_controls: {
    separator_pressure: number;
    separator_temp: number;
    gas_fraction: number;
  };
  best_cost: number;
  best_metrics: {
    vapor_fraction?: number;
    gas_flow_rate?: number;
    liquid_flow_rate?: number;
  } | null;
}

export interface FacilityOptimizeResponse {
  message: string;
  history: FacilityOptimizeHistoryEntry[];
  result: FacilityOptimizeResult;
  pipe_context: { outlet_pressure: number; temperature: number; throughput: number };
  timestamp: string;
}

export interface FacilityOptimizeStatus {
  history: FacilityOptimizeHistoryEntry[];
  result: FacilityOptimizeResult | null;
  pipe_context?: { outlet_pressure: number; temperature: number; throughput: number };
  timestamp?: string;
}

export type BackendStreamMessage =
  | { type: 'iteration'; payload: IterationPayload }
  | { type: 'complete'; payload: OptimizationResult };

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  private readonly apiUrl = 'http://localhost:8000';
  private readonly wsUrl = 'ws://localhost:8000/ws/updates';
  private stream$?: WebSocketSubject<BackendStreamMessage>;

  constructor(private readonly http: HttpClient) {}

  simulateAndTrain(numSystems: number, seed?: number): Observable<SimulateResponse> {
    return this.http.post<SimulateResponse>(`${this.apiUrl}/simulate-train`, { num_systems: numSystems, seed });
  }

  startOptimization(body: {
    max_iters: number;
    population: number;
    initial_D?: number;
    initial_V?: number;
    target_outlet_pressure?: number;
    target_mean_pressure?: number;
    target_gradient?: number;
  }): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.apiUrl}/optimize`, body);
  }

  stopOptimization(): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.apiUrl}/optimize/stop`, {});
  }

  fetchStatus(): Observable<{ running: boolean; iterations: IterationPayload[]; result: OptimizationResult | null }> {
    return this.http.get<{ running: boolean; iterations: IterationPayload[]; result: OptimizationResult | null }>(
      `${this.apiUrl}/status`,
    );
  }

  fetchResult(): Observable<OptimizationResult> {
    return this.http.get<OptimizationResult>(`${this.apiUrl}/result`);
  }

  fetchOptimizationHistory(): Observable<OptimizationHistoryResponse> {
    return this.http.get<OptimizationHistoryResponse>(`${this.apiUrl}/optimize/history`);
  }

  fetchSurrogateStatus(): Observable<SurrogateStatus> {
    return this.http.get<SurrogateStatus>(`${this.apiUrl}/surrogate/status`);
  }

  fetchSurrogateHistory(): Observable<{ training_history: TrainingHistoryEntry[] }> {
    return this.http.get<{ training_history: TrainingHistoryEntry[] }>(`${this.apiUrl}/surrogate/history`);
  }

  facilitySimulate(numSamples: number, seed?: number): Observable<{ message: string; dataset_size: number }> {
    return this.http.post<{ message: string; dataset_size: number }>(`${this.apiUrl}/facility/simulate`, {
      num_samples: numSamples,
      seed,
    });
  }

  facilityTrain(payload: { epochs: number; batch_size: number; learning_rate: number; val_split: number }): Observable<any> {
    return this.http.post(`${this.apiUrl}/facility/train`, payload);
  }

  fetchFacilityStatus(): Observable<FacilityStatus> {
    return this.http.get<FacilityStatus>(`${this.apiUrl}/facility/status`);
  }

  runPipeline(body: {
    D: number;
    v: number;
    t: number;
    separator_pressure: number;
    separator_temp: number;
    gas_fraction: number;
  }): Observable<PipelineRunResponse> {
    return this.http.post<PipelineRunResponse>(`${this.apiUrl}/pipeline/run`, body);
  }

  optimizeFacility(body: {
    max_iters: number;
    population: number;
    initial_separator_pressure?: number;
    initial_separator_temp?: number;
    initial_gas_fraction?: number;
    target_vapor_fraction?: number | null;
    target_gas_flow_rate?: number | null;
    target_liquid_flow_rate?: number | null;
    weight_vapor?: number;
    weight_gas?: number;
    weight_liquid?: number;
    pipe_D?: number;
    pipe_v?: number;
    pipe_t?: number;
  }): Observable<FacilityOptimizeResponse> {
    return this.http.post<FacilityOptimizeResponse>(`${this.apiUrl}/facility/optimize`, body);
  }

  fetchFacilityOptimizationStatus(): Observable<FacilityOptimizeStatus> {
    return this.http.get<FacilityOptimizeStatus>(`${this.apiUrl}/facility/optimize/status`);
  }

  connectStream(): Observable<BackendStreamMessage> {
    if (!this.stream$ || this.stream$.closed) {
      this.stream$ = webSocket<BackendStreamMessage>(this.wsUrl);
    }
    return this.stream$;
  }

  closeStream(): void {
    this.stream$?.complete();
    this.stream$ = undefined;
  }
}
