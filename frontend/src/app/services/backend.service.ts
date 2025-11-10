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
  grad_D?: number;
  grad_V?: number;
  grad_norm?: number;
}

export interface OptimizationResult {
  best_params: { D: number; v: number };
  best_cost: number;
  best_profile: number[];
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
  };
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

  startOptimization(maxIters: number, population: number, initialD: number, initialV: number): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.apiUrl}/optimize`, {
      max_iters: maxIters,
      population,
      initial_D: initialD,
      initial_V: initialV,
    });
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
