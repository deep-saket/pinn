import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import Chart from 'chart.js/auto';

import {
  BackendService,
  BackendStreamMessage,
  IterationPayload,
  OptimizationHistoryResponse,
  OptimizationResult,
  SimulateResponse,
  SurrogateStatus,
  TrainingHistoryEntry,
} from '../../services/backend.service';
import { ResultPanelComponent } from '../result-panel/result-panel.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule, ResultPanelComponent],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('pressureCanvas') pressureCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('costCanvas') costCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('lossCanvas') lossCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('gradCanvas') gradCanvas?: ElementRef<HTMLCanvasElement>;

  controls = {
    numSystems: 10,
    seed: undefined as number | undefined,
    maxIters: 30,
    population: 24,
    initialD: 0.08,
    initialV: 0.5,
  };

  surrogateReady = false;
  running = false;
  darkMode = false;
  latestIteration?: IterationPayload;
  finalResult: OptimizationResult | null = null;
  xGrid: number[] = [];
  targetProfile: number[] = [];
  currentProfile: number[] = [];
  surrogateStatus?: SurrogateStatus;
  trainingHistory: TrainingHistoryEntry[] = [];
  statusLog: string[] = [];
  errorMessage = '';

  private pressureChart?: Chart;
  private costChart?: Chart;
  private lossChart?: Chart;
  private gradChart?: Chart;
  private costSeries: number[] = [];
  private iterationLabels: string[] = [];
  private gradLabels: string[] = [];
  private gradDSeries: (number | null)[] = [];
  private gradVSeries: (number | null)[] = [];
  private streamSub?: Subscription;

  constructor(private readonly backend: BackendService) {}

  ngOnInit(): void {
    this.streamSub = this.backend.connectStream().subscribe({
      next: (message) => this.handleStreamMessage(message),
      error: () => this.logMessage('WebSocket connection closed.'),
    });
    this.refreshSurrogateStatus();
    this.loadOptimizationHistory();
  }

  ngAfterViewInit(): void {
    this.initCharts();
    this.applyThemeToCharts();
    this.updateBodyTheme();
  }

  ngOnDestroy(): void {
    this.streamSub?.unsubscribe();
    this.backend.closeStream();
    this.pressureChart?.destroy();
    this.costChart?.destroy();
    document.body.classList.remove('dark-theme');
  }

  simulateAndTrain(): void {
    this.errorMessage = '';
    this.backend
      .simulateAndTrain(this.controls.numSystems, this.controls.seed)
      .subscribe({
        next: (response: SimulateResponse) => {
          this.handleTrainingResponse(response);
          this.logMessage(`Surrogate trained with ${response.dataset_size} samples.`);
        },
        error: () => {
          this.errorMessage = 'Failed to run simulate & train.';
        },
      });
  }

  startOptimization(): void {
    if (!this.surrogateReady) {
      this.errorMessage = 'Train the surrogate first.';
      return;
    }
    this.errorMessage = '';
    this.costSeries = [];
    this.iterationLabels = [];
    this.gradLabels = [];
    this.gradDSeries = [];
    this.gradVSeries = [];
    this.latestIteration = undefined;
    this.finalResult = null;
    this.costChart?.update();
    this.gradChart?.update();
    this.backend
      .startOptimization(
        this.controls.maxIters,
        this.controls.population,
        this.controls.initialD,
        this.controls.initialV,
      )
      .subscribe({
        next: () => {
          this.running = true;
          this.logMessage('Optimization started.');
        },
        error: () => {
          this.errorMessage = 'Unable to start optimization.';
        },
      });
  }

  stopOptimization(): void {
    if (!this.running) {
      return;
    }
    this.backend.stopOptimization().subscribe({
      next: () => {
        this.logMessage('Stop requested.');
      },
    });
  }

  toggleDarkMode(): void {
    this.darkMode = !this.darkMode;
    this.applyThemeToCharts();
    this.updateBodyTheme();
  }

  refreshSurrogateStatus(): void {
    this.backend.fetchSurrogateStatus().subscribe({
      next: (status) => {
        this.surrogateStatus = status;
        this.surrogateReady = status.has_model;
        if (status.training_history?.length) {
          this.trainingHistory = status.training_history;
          this.updateLossChart();
        }
        if (status.sample_prediction) {
          this.xGrid = status.sample_prediction.x;
          this.currentProfile = status.sample_prediction.u_hat;
          if (status.sample_prediction.u_hat?.length) {
            this.updatePressureChart(status.sample_prediction.u_hat);
          }
        }
      },
      error: () => {
        this.logMessage('Unable to fetch surrogate status.');
      },
    });
  }

  loadOptimizationHistory(): void {
    this.backend.fetchOptimizationHistory().subscribe({
      next: (response: OptimizationHistoryResponse) => {
        if (response.history?.length) {
          this.applyOptimizationHistory(response);
          this.logMessage('Loaded latest optimization history.');
        }
      },
      error: () => {
        this.logMessage('No saved optimization history found.');
      },
    });
  }

  private initCharts(): void {
    if (this.pressureCanvas && !this.pressureChart) {
      this.pressureChart = new Chart(this.pressureCanvas.nativeElement, {
        type: 'line',
        data: {
          labels: this.xGrid,
          datasets: [
            {
              label: 'Target profile',
              data: [],
              borderColor: '#f94144',
              borderDash: [6, 3],
              tension: 0.2,
            },
            {
              label: 'Surrogate prediction',
              data: [],
              borderColor: '#0077b6',
              tension: 0.2,
            },
            {
              label: 'Optimizer best profile',
              data: [],
              borderColor: '#2a9d8f',
              borderDash: [3, 3],
              tension: 0.2,
            },
          ],
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'x' } },
            y: { title: { display: true, text: 'u(x)' } },
          },
        },
      });
    }

    if (this.costCanvas && !this.costChart) {
      this.costChart = new Chart(this.costCanvas.nativeElement, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Cost',
              data: [],
              borderColor: '#06d6a0',
              tension: 0.2,
            },
          ],
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: 'Iteration' },
              ticks: { maxTicksLimit: 8, autoSkip: true },
            },
            y: { title: { display: true, text: 'J(D, v)' } },
          },
        },
      });
    }

    if (this.lossCanvas && !this.lossChart) {
      this.lossChart = new Chart(this.lossCanvas.nativeElement, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Train Loss',
              data: [],
              borderColor: '#1d3557',
              tension: 0.2,
            },
            {
              label: 'Validation Loss',
              data: [],
              borderColor: '#ffb703',
              borderDash: [4, 2],
              tension: 0.2,
            },
          ],
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: 'Epoch' },
              ticks: { maxTicksLimit: 10, autoSkip: true },
            },
            y: { title: { display: true, text: 'Loss' } },
          },
        },
      });
    }

    if (this.gradCanvas && !this.gradChart) {
      this.gradChart = new Chart(this.gradCanvas.nativeElement, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: '∂J/∂D',
              data: [],
              borderColor: '#7209b7',
              tension: 0.2,
            },
            {
              label: '∂J/∂v',
              data: [],
              borderColor: '#faa307',
              tension: 0.2,
            },
          ],
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: 'Iteration' },
              ticks: { maxTicksLimit: 10, autoSkip: true },
            },
            y: { title: { display: true, text: 'Gradient' } },
          },
        },
      });
    }

  }

  private updatePressureChart(currentProfile: number[]): void {
    if (!this.pressureChart) {
      this.initCharts();
    }
    if (!this.pressureChart) {
      return;
    }
    this.pressureChart.data.labels = this.xGrid;
    this.pressureChart.data.datasets[0].data = this.targetProfile ?? [];
    this.pressureChart.data.datasets[1].data = currentProfile;
    this.pressureChart.data.datasets[2].data = this.currentProfile ?? [];
    this.pressureChart.update();
    this.currentProfile = currentProfile;
  }

  private updateLossChart(): void {
    if (!this.lossChart) {
      this.initCharts();
    }
    if (!this.lossChart) {
      return;
    }
    const history = this.trainingHistory;
    this.lossChart.data.labels = history.map((entry) => entry.epoch);
    this.lossChart.data.datasets[0].data = history.map((entry) => entry.train_total_loss ?? null);
    this.lossChart.data.datasets[1].data = history.map((entry) => entry.val_total_loss ?? null);
    this.lossChart.update();
  }

  private updateCostChart(iteration: number, cost: number): void {
    if (!this.costChart) {
      this.initCharts();
    }
    if (!this.costChart) {
      return;
    }
    this.iterationLabels.push(iteration.toString());
    this.costSeries.push(cost);
    this.costChart.data.labels = this.iterationLabels;
    this.costChart.data.datasets[0].data = this.costSeries;
    this.costChart.update();
  }

  private updateGradientChart(iteration: number, gradD?: number, gradV?: number): void {
    if (!this.gradChart) {
      this.initCharts();
    }
    if (!this.gradChart) {
      return;
    }
    this.gradLabels.push(iteration.toString());
    this.gradDSeries.push(gradD ?? null);
    this.gradVSeries.push(gradV ?? null);
    this.gradChart.data.labels = this.gradLabels;
    this.gradChart.data.datasets[0].data = this.gradDSeries;
    this.gradChart.data.datasets[1].data = this.gradVSeries;
    this.gradChart.update();
  }

  private handleStreamMessage(message: BackendStreamMessage): void {
    if (message.type === 'iteration') {
      this.running = true;
      this.latestIteration = message.payload;
      this.updatePressureChart(message.payload.u_profile);
      this.updateCostChart(message.payload.iteration, message.payload.cost);
      this.updateGradientChart(message.payload.iteration, message.payload.grad_D, message.payload.grad_V);
      this.logMessage(
        `Iter ${message.payload.iteration}: cost=${message.payload.cost.toFixed(4)} D=${message.payload.D.toFixed(
          3,
        )} v=${message.payload.v.toFixed(3)}`,
      );
    } else if (message.type === 'complete') {
      this.running = false;
      this.finalResult = message.payload;
      this.logMessage('Optimization completed.');
    }
    this.applyThemeToCharts();
  }

  private logMessage(message: string): void {
    const timestamp = new Date().toLocaleTimeString();
    this.statusLog = [`[${timestamp}] ${message}`, ...this.statusLog].slice(0, 15);
  }

  private handleTrainingResponse(response: SimulateResponse): void {
    this.surrogateReady = true;
    this.xGrid = response.x_grid;
    this.targetProfile = response.target_profile;
    if (response.training_history?.length) {
      this.trainingHistory = response.training_history;
      this.updateLossChart();
    }
    this.refreshSurrogateStatus();
  }

  private applyOptimizationHistory(response: OptimizationHistoryResponse): void {
    const iterations = response.history ?? [];
    this.costSeries = [];
    this.iterationLabels = [];
    iterations.forEach((record) => {
      this.iterationLabels.push(record.iteration.toString());
      this.costSeries.push(record.cost);
      this.gradLabels.push(record.iteration.toString());
      this.gradDSeries.push(record.grad_D ?? null);
      this.gradVSeries.push(record.grad_V ?? null);
    });
    if (this.costChart) {
      this.costChart.data.labels = this.iterationLabels;
      this.costChart.data.datasets[0].data = this.costSeries;
      this.costChart.update();
    }
    if (this.gradChart) {
      this.gradChart.data.labels = this.gradLabels;
      this.gradChart.data.datasets[0].data = this.gradDSeries;
      this.gradChart.data.datasets[1].data = this.gradVSeries;
      this.gradChart.update();
    }

    if (response.result) {
      this.finalResult = response.result;
    }
    if (iterations.length) {
      this.latestIteration = iterations[iterations.length - 1];
    }
    if (iterations.length) {
      const lastProfile = iterations[iterations.length - 1].u_profile;
      if (lastProfile?.length) {
        this.currentProfile = lastProfile;
        this.updatePressureChart(lastProfile);
      }
    }
  }

  hasGradientSamples(): boolean {
    return this.gradLabels.length > 0;
  }

  private applyThemeToCharts(): void {
    const axisColor = this.darkMode ? '#e2e8f0' : '#1f2933';
    const gridColor = this.darkMode ? 'rgba(226,232,240,0.2)' : 'rgba(148,163,184,0.3)';

    const applyScales = (chart?: Chart) => {
      if (!chart || !chart.options.scales) {
        return;
      }
      const scales = chart.options.scales as Record<string, any>;
      Object.values(scales).forEach((scale: any) => {
        scale.ticks = { ...(scale.ticks || {}), color: axisColor };
        scale.grid = { ...(scale.grid || {}), color: gridColor };
      });
      chart.update('none');
    };

    const updatePressureColors = () => {
      if (!this.pressureChart) {
        return;
      }
      const datasets = this.pressureChart.data.datasets;
      if (datasets[0]) {
        datasets[0].borderColor = this.darkMode ? '#f87171' : '#f94144';
      }
      if (datasets[1]) {
        datasets[1].borderColor = this.darkMode ? '#38bdf8' : '#0077b6';
      }
      if (datasets[2]) {
        datasets[2].borderColor = this.darkMode ? '#5eead4' : '#2a9d8f';
      }
      this.pressureChart.update('none');
    };

    const updateCostColors = () => {
      if (!this.costChart) {
        return;
      }
      const dataset = this.costChart.data.datasets[0];
      if (dataset) {
        dataset.borderColor = this.darkMode ? '#c084fc' : '#06d6a0';
      }
      this.costChart.update('none');
    };

    const updateLossColors = () => {
      if (!this.lossChart) {
        return;
      }
      const ds = this.lossChart.data.datasets;
      if (ds[0]) {
        ds[0].borderColor = this.darkMode ? '#f59e0b' : '#1d3557';
      }
      if (ds[1]) {
        ds[1].borderColor = this.darkMode ? '#2dd4bf' : '#ffb703';
      }
      this.lossChart.update('none');
    };

    const updateGradColors = () => {
      if (!this.gradChart) {
        return;
      }
      const ds = this.gradChart.data.datasets;
      if (ds[0]) {
        ds[0].borderColor = this.darkMode ? '#c77dff' : '#7209b7';
      }
      if (ds[1]) {
        ds[1].borderColor = this.darkMode ? '#f97316' : '#faa307';
      }
      this.gradChart.update('none');
    };

    applyScales(this.pressureChart);
    applyScales(this.costChart);
    applyScales(this.lossChart);
    applyScales(this.gradChart);
    updatePressureColors();
    updateCostColors();
    updateLossColors();
    updateGradColors();
  }

  private updateBodyTheme(): void {
    document.body.classList.toggle('dark-theme', this.darkMode);
  }
}
