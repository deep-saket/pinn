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
  FacilityOptimizeHistoryEntry,
  FacilityOptimizeResult,
  FacilityOptimizeStatus,
  FacilityStatus,
  FinancialOptimizeHistoryEntry,
  FinancialOptimizeResult,
  FinancialOptimizeStatus,
  FinancialStatus,
  IterationPayload,
  OptimizationHistoryResponse,
  OptimizationResult,
  PipelineRunResponse,
  SimulateResponse,
  SurrogateStatus,
  TrainingHistoryEntry,
} from '../../services/backend.service';

type FacilityHistoryEntry = {
  epoch: number;
  train_loss?: number | null;
  train_total_loss?: number | null;
  val_loss?: number | null;
};
@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('pressureCanvas') pressureCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('costCanvas') costCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('lossCanvas') lossCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('gradCanvas') gradCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('spatialGradientCanvas') spatialGradientCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('statusPanel') statusPanel?: ElementRef<HTMLDivElement>;

  pipelineControls = {
    separatorPressure: 100,
    separatorTemp: 45,
    gasFraction: 0.5,
  };

  financialControls = {
    pricePerUnit: 80,
    hedgeRatio: 0.5,
    opexMultiplier: 1.0,
  };

  pipeTargets = {
    outletPressure: 185,
    meanPressure: 150,
    outletGradient: 0.0,
  };

  surrogateSimControls = {
    numSystems: 12,
    seed: undefined as number | undefined,
  };

  optimizationControls = {
    maxIters: 30,
    population: 24,
  };

  surrogateReady = false;
  running = false;
  darkMode = false;
  latestIteration?: IterationPayload;
  finalResult: OptimizationResult | null = null;
  xGrid: number[] = [];
  targetProfile: number[] = [];
  optimizerProfile: number[] = [];
  currentProfile: number[] = [];
  surrogatePreviewProfile: number[] = [];
  targetGradientProfile: number[] = [];
  optimizerGradientProfile: number[] = [];
  surrogateGradientPreview: number[] = [];
  surrogateStatus?: SurrogateStatus;
  facilityStatus?: FacilityStatus;
  financialStatus?: FinancialStatus;
  trainingHistory: TrainingHistoryEntry[] = [];
  facilityTrainingHistory: FacilityHistoryEntry[] = [];
  financialTrainingHistory: FacilityHistoryEntry[] = [];
  facilityOptimizationHistory: FacilityOptimizeHistoryEntry[] = [];
  facilityOptimizationResult: FacilityOptimizeResult | null = null;
  facilityOptimizationTimestamp: string | null = null;
  facilityPipeContext: { outlet_pressure: number; temperature: number; throughput: number } | null = null;
  financialOptimizationHistory: FinancialOptimizeHistoryEntry[] = [];
  financialOptimizationResult: FinancialOptimizeResult | null = null;
  financialOptimizationTimestamp: string | null = null;
  financialFacilityContext: { vapor_fraction: number; gas_flow_rate: number; liquid_flow_rate: number } | null = null;
  statusLog: string[] = [];
  errorMessage = '';
  facilityError = '';
  facilityOptError = '';
  financialError = '';
  financialOptError = '';
  facilityTraining = {
    numSamples: 2000,
    epochs: 150,
    batchSize: 256,
    learningRate: 1e-3,
    valSplit: 0.2,
  };
  financialTraining = {
    numSamples: 2000,
    epochs: 150,
    batchSize: 256,
    learningRate: 1e-3,
    valSplit: 0.2,
  };
  facilityOptimizationControls = {
    maxIters: 20,
    population: 20,
  };
  facilityObjectives = {
    vaporFraction: 0.65,
    gasFlow: 18,
    liquidFlow: 20,
  };
  facilityOptimizerBusy = false;
  financialOptimizationControls = {
    maxIters: 20,
    population: 20,
  };
  financialObjectives = {
    netProfit: 800,
    riskIndex: 5,
  };
  financialOptimizerBusy = false;
  pipelinePreview: PipelineRunResponse | null = null;
  stageView: 'pipe' | 'facility' | 'financial' = 'pipe';
  statusOverlay: { stage: 'pipe' | 'facility' | 'financial'; x: number; y: number } | null = null;

  setStage(view: 'pipe' | 'facility' | 'financial'): void {
    if (this.stageView === view) {
      return;
    }
    this.stageView = view;
    this.statusOverlay = null;
    this.refreshCharts();
  }

  private refreshCharts(): void {
    this.updatePressureChart();
    this.updateSpatialGradientChart();
    this.updateCostChart();
    this.updateGradientChart();
  }

  private pressureChart?: Chart;
  private costChart?: Chart;
  private lossChart?: Chart;
  private gradChart?: Chart;
  private gradProfileChart?: Chart;
  private costSeries: number[] = [];
  private iterationLabels: string[] = [];
  private gradLabels: string[] = [];
  private gradDSeries: (number | null)[] = [];
  private gradVSeries: (number | null)[] = [];
  private streamSub?: Subscription;
  private readonly defaultPipeGuess = { D: 0.08, v: 0.5 };

  constructor(private readonly backend: BackendService) {}

  ngOnInit(): void {
    this.streamSub = this.backend.connectStream().subscribe({
      next: (message) => this.handleStreamMessage(message),
      error: () => this.logMessage('WebSocket connection closed.'),
    });
    this.refreshSurrogateStatus();
    this.refreshFacilityStatus();
    this.refreshFacilityOptimizationStatus();
    this.refreshFinancialStatus();
    this.refreshFinancialOptimizationStatus();
    this.loadOptimizationHistory();
  }

  ngAfterViewInit(): void {
    this.initCharts();
    this.applyThemeToCharts();
    this.updateBodyTheme();
    this.updatePressureChart();
    this.updateSpatialGradientChart();
    this.updateLossChart();
    this.costChart?.update();
    this.gradChart?.update();
  }

  ngOnDestroy(): void {
    this.streamSub?.unsubscribe();
    this.backend.closeStream();
    this.pressureChart?.destroy();
    this.costChart?.destroy();
    this.lossChart?.destroy();
    this.gradChart?.destroy();
    this.gradProfileChart?.destroy();
    document.body.classList.remove('dark-theme');
  }

  simulateAndTrain(): void {
    this.errorMessage = '';
    this.backend
      .simulateAndTrain(this.surrogateSimControls.numSystems, this.surrogateSimControls.seed)
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
    this.optimizerProfile = [];
    this.optimizerGradientProfile = [];
    this.updatePressureChart();
    this.updateSpatialGradientChart();
    this.latestIteration = undefined;
    this.finalResult = null;
    this.costChart?.update();
    this.gradChart?.update();
    this.backend
      .startOptimization({
        max_iters: this.optimizationControls.maxIters,
        population: this.optimizationControls.population,
        target_outlet_pressure: this.pipeTargets.outletPressure,
        target_mean_pressure: this.pipeTargets.meanPressure,
        target_gradient: this.pipeTargets.outletGradient,
        initial_D: this.defaultPipeGuess.D,
        initial_V: this.defaultPipeGuess.v,
      })
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
          if (status.sample_prediction.x?.length) {
            this.xGrid = status.sample_prediction.x;
            this.targetGradientProfile = this.computeGradientFromProfile(this.targetProfile);
          }
          this.surrogatePreviewProfile = status.sample_prediction.u_hat ?? [];
          this.surrogateGradientPreview = status.sample_prediction.du_dx ?? [];
          this.updatePressureChart();
          this.updateSpatialGradientChart();
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

  simulateFacilityDataset(): void {
    this.facilityError = '';
    this.backend
      .facilitySimulate(this.facilityTraining.numSamples)
      .subscribe({
        next: (response) => {
          this.logMessage(`Facility dataset generated with ${response.dataset_size} samples.`);
          this.refreshFacilityStatus();
        },
        error: () => {
          this.facilityError = 'Failed to simulate facility dataset.';
        },
      });
  }

  trainFacilitySurrogate(): void {
    this.facilityError = '';
    this.backend
      .facilityTrain({
        epochs: this.facilityTraining.epochs,
        batch_size: this.facilityTraining.batchSize,
        learning_rate: this.facilityTraining.learningRate,
        val_split: this.facilityTraining.valSplit,
      })
      .subscribe({
        next: () => {
          this.logMessage('Facility surrogate trained.');
          this.refreshFacilityStatus();
        },
        error: () => {
          this.facilityError = 'Failed to train facility surrogate.';
        },
      });
  }

  simulateFinancialDataset(): void {
    this.financialError = '';
    this.backend.financialSimulate(this.financialTraining.numSamples).subscribe({
      next: (response) => {
        this.logMessage(`Financial dataset generated with ${response.dataset_size} samples.`);
        this.refreshFinancialStatus();
      },
      error: () => {
        this.financialError = 'Failed to simulate financial dataset.';
      },
    });
  }

  trainFinancialSurrogate(): void {
    this.financialError = '';
    this.backend
      .financialTrain({
        epochs: this.financialTraining.epochs,
        batch_size: this.financialTraining.batchSize,
        learning_rate: this.financialTraining.learningRate,
        val_split: this.financialTraining.valSplit,
      })
      .subscribe({
        next: () => {
          this.logMessage('Financial surrogate trained.');
          this.refreshFinancialStatus();
        },
        error: () => {
          this.financialError = 'Failed to train financial surrogate.';
        },
      });
  }

  refreshFacilityStatus(): void {
    this.backend.fetchFacilityStatus().subscribe({
      next: (status) => {
        this.facilityStatus = status;
        const rawHistory = status.training_history ?? [];
        this.facilityTrainingHistory = rawHistory.map((entry: any) => ({
          epoch: entry.epoch,
          train_loss: entry.train_loss ?? entry.train_total_loss ?? entry.total_loss ?? null,
          val_loss: entry.val_loss ?? entry.val_total_loss ?? null,
        }));
        if (this.stageView === 'facility') {
          this.updateCostChart();
          this.updateGradientChart();
        }
      },
      error: () => {
        this.logMessage('Unable to fetch facility status.');
      },
    });
  }

  refreshFacilityOptimizationStatus(): void {
    this.backend.fetchFacilityOptimizationStatus().subscribe({
      next: (status: FacilityOptimizeStatus) => {
        this.facilityOptimizationHistory = status.history ?? [];
        this.facilityOptimizationResult = status.result ?? null;
        this.facilityOptimizationTimestamp = status.timestamp ?? null;
        this.facilityPipeContext = status.pipe_context ?? null;
        this.updateCostChart();
        this.updateGradientChart();
      },
      error: () => {
        this.logMessage('Unable to fetch facility optimization status.');
      },
    });
  }

  refreshStageStatus(): void {
    if (this.stageView === 'pipe') {
      this.refreshSurrogateStatus();
    } else if (this.stageView === 'facility') {
      this.refreshFacilityStatus();
      this.refreshFacilityOptimizationStatus();
    } else {
      this.refreshFinancialStatus();
      this.refreshFinancialOptimizationStatus();
    }
  }

  refreshFinancialStatus(): void {
    this.backend.fetchFinancialStatus().subscribe({
      next: (status) => {
        this.financialStatus = status;
        const rawHistory = status.training_history ?? [];
        this.financialTrainingHistory = rawHistory.map((entry: any) => ({
          epoch: entry.epoch,
          train_loss: entry.train_loss ?? entry.train_total_loss ?? entry.total_loss ?? null,
          val_loss: entry.val_loss ?? entry.val_total_loss ?? null,
        }));
        if (this.stageView === 'financial') {
          this.updateCostChart();
          this.updateGradientChart();
        }
      },
      error: () => {
        this.logMessage('Unable to fetch financial status.');
      },
    });
  }

  refreshFinancialOptimizationStatus(): void {
    this.backend.fetchFinancialOptimizationStatus().subscribe({
      next: (status: FinancialOptimizeStatus) => {
        this.financialOptimizationHistory = status.history ?? [];
        this.financialOptimizationResult = status.result ?? null;
        this.financialOptimizationTimestamp = status.timestamp ?? null;
        this.financialFacilityContext = status.facility_context ?? null;
        this.updateCostChart();
        this.updateGradientChart();
      },
      error: () => {
        this.logMessage('Unable to fetch financial optimization status.');
      },
    });
  }

  runPipeline(): void {
    this.facilityError = '';
    this.financialError = '';
    const pipeInputs = this.resolvePipeInputs();
    this.backend
      .runPipeline({
        D: pipeInputs.D,
        v: pipeInputs.v,
        t: this.stateTargetTime(),
        separator_pressure: this.pipelineControls.separatorPressure,
        separator_temp: this.pipelineControls.separatorTemp,
        gas_fraction: this.pipelineControls.gasFraction,
        price_per_unit: this.financialControls.pricePerUnit,
        hedge_ratio: this.financialControls.hedgeRatio,
        opex_multiplier: this.financialControls.opexMultiplier,
      })
      .subscribe({
        next: (response) => {
          this.pipelinePreview = response;
          this.logMessage('Pipeline evaluated.');
          if (response.pipe?.profile) {
            this.optimizerProfile = response.pipe.profile;
          }
          this.refreshCharts();
        },
        error: () => {
          this.facilityError = 'Pipeline run failed. Ensure surrogates are trained.';
          this.financialError = 'Pipeline run failed. Ensure surrogates are trained.';
        },
      });
  }

  optimizeFacility(): void {
    if (this.facilityOptimizerBusy) {
      return;
    }
    this.facilityOptError = '';
    this.facilityOptimizerBusy = true;
    const pipeInputs = this.resolvePipeInputs();
    this.backend
      .optimizeFacility({
        max_iters: this.facilityOptimizationControls.maxIters,
        population: this.facilityOptimizationControls.population,
        initial_separator_pressure: this.pipelineControls.separatorPressure,
        initial_separator_temp: this.pipelineControls.separatorTemp,
        initial_gas_fraction: this.pipelineControls.gasFraction,
        target_vapor_fraction: this.facilityObjectives.vaporFraction,
        target_gas_flow_rate: this.facilityObjectives.gasFlow,
        target_liquid_flow_rate: this.facilityObjectives.liquidFlow,
        pipe_D: pipeInputs.D,
        pipe_v: pipeInputs.v,
        pipe_t: this.stateTargetTime(),
      })
      .subscribe({
        next: (response) => {
          this.facilityOptimizationHistory = response.history;
          this.facilityOptimizationResult = response.result;
          this.facilityOptimizationTimestamp = response.timestamp;
          this.facilityPipeContext = response.pipe_context;
          this.facilityOptimizerBusy = false;
          this.logMessage('Facility optimization complete.');
          this.updateCostChart();
          this.updateGradientChart();
        },
        error: () => {
          this.facilityOptimizerBusy = false;
          this.facilityOptError = 'Failed to optimize facility.';
        },
      });
  }

  optimizeFinancial(): void {
    if (this.financialOptimizerBusy) {
      return;
    }
    this.financialOptError = '';
    this.financialOptimizerBusy = true;
    const pipeInputs = this.resolvePipeInputs();
    this.backend
      .optimizeFinancial({
        max_iters: this.financialOptimizationControls.maxIters,
        population: this.financialOptimizationControls.population,
        initial_price: this.financialControls.pricePerUnit,
        initial_hedge: this.financialControls.hedgeRatio,
        initial_opex: this.financialControls.opexMultiplier,
        target_net_profit: this.financialObjectives.netProfit,
        target_risk_index: this.financialObjectives.riskIndex,
        weight_profit: 1.0,
        weight_risk: 0.6,
        pipe_D: pipeInputs.D,
        pipe_v: pipeInputs.v,
        pipe_t: this.stateTargetTime(),
        separator_pressure: this.pipelineControls.separatorPressure,
        separator_temp: this.pipelineControls.separatorTemp,
        gas_fraction: this.pipelineControls.gasFraction,
      })
      .subscribe({
        next: (response) => {
          this.financialOptimizationHistory = response.history;
          this.financialOptimizationResult = response.result;
          this.financialOptimizationTimestamp = response.timestamp;
          this.financialFacilityContext = response.facility_context;
          this.financialOptimizerBusy = false;
          this.logMessage('Financial optimization complete.');
          this.updateCostChart();
          this.updateGradientChart();
        },
        error: () => {
          this.financialOptimizerBusy = false;
          this.financialOptError = 'Failed to optimize financial stage.';
        },
      });
  }

  private stateTargetTime(): number {
    if (this.surrogateStatus?.sample_prediction?.t !== undefined) {
      return this.surrogateStatus.sample_prediction.t;
    }
    return 0.6;
  }

  private resolvePipeInputs(): { D: number; v: number } {
    if (this.finalResult?.best_params) {
      return {
        D: this.finalResult.best_params.D,
        v: this.finalResult.best_params.v,
      };
    }
    const preview = this.surrogateStatus?.sample_prediction;
    if (preview) {
      return {
        D: preview.D ?? this.defaultPipeGuess.D,
        v: preview.v ?? this.defaultPipeGuess.v,
      };
    }
    return { ...this.defaultPipeGuess };
  }

  get pipePreviewParams(): { D: number; v: number } {
    return this.resolvePipeInputs();
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
            {
              label: '',
              data: [],
              borderColor: '#2563eb',
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
            {
              label: '∂J/∂Gas',
              data: [],
              borderColor: '#0ea5e9',
              borderDash: [4, 3],
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

    if (this.spatialGradientCanvas && !this.gradProfileChart) {
      this.gradProfileChart = new Chart(this.spatialGradientCanvas.nativeElement, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: '∂u/∂x target',
              data: [],
              borderColor: '#ef476f',
              borderDash: [6, 3],
              tension: 0.2,
            },
            {
              label: '∂u/∂x surrogate',
              data: [],
              borderColor: '#118ab2',
              tension: 0.2,
            },
            {
              label: '∂u/∂x optimizer',
              data: [],
              borderColor: '#06d6a0',
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
            x: { title: { display: true, text: 'x' } },
            y: { title: { display: true, text: '∂u/∂x' } },
          },
        },
      });
    }

  }

  private updatePressureChart(currentProfile?: number[]): void {
    if (!this.pressureChart) {
      this.initCharts();
    }
    if (!this.pressureChart) {
      return;
    }
    if (this.stageView === 'pipe') {
      if (currentProfile?.length) {
        this.currentProfile = currentProfile;
      }
      const preview = this.surrogatePreviewProfile ?? [];
      const optimizerData = this.optimizerProfile ?? [];
      this.pressureChart.data.labels = this.xGrid;
      this.pressureChart.data.datasets[0].label = 'Target profile';
      this.pressureChart.data.datasets[0].data = this.targetProfile ?? [];
      if (this.pressureChart.data.datasets[1]) {
        this.pressureChart.data.datasets[1].label = 'Surrogate preview';
        this.pressureChart.data.datasets[1].data = preview;
      }
      if (this.pressureChart.data.datasets[2]) {
        this.pressureChart.data.datasets[2].label = 'Optimizer profile';
        this.pressureChart.data.datasets[2].data = this.currentProfile ?? optimizerData;
      }
    } else if (this.stageView === 'facility') {
      const facility = this.pipelinePreview?.facility;
      const labels = ['Vapor Fraction', 'Gas Flow', 'Liquid Flow'];
      const values = [
        facility?.vapor_fraction ?? 0,
        facility?.gas_flow_rate ?? 0,
        facility?.liquid_flow_rate ?? 0,
      ];
      this.pressureChart.data.labels = labels;
      this.pressureChart.data.datasets[0].label = 'Facility outputs';
      this.pressureChart.data.datasets[0].data = values;
      if (this.pressureChart.data.datasets[1]) {
        this.pressureChart.data.datasets[1].label = '';
        this.pressureChart.data.datasets[1].data = [];
      }
      if (this.pressureChart.data.datasets[2]) {
        this.pressureChart.data.datasets[2].label = '';
        this.pressureChart.data.datasets[2].data = [];
      }
    } else {
      const financial = this.pipelinePreview?.financial;
      const labels = ['Net Profit', 'Risk Index'];
      const values = [financial?.net_profit ?? 0, financial?.risk_index ?? 0];
      this.pressureChart.data.labels = labels;
      this.pressureChart.data.datasets[0].label = 'Financial metrics';
      this.pressureChart.data.datasets[0].data = values;
      if (this.pressureChart.data.datasets[1]) {
        this.pressureChart.data.datasets[1].label = '';
        this.pressureChart.data.datasets[1].data = [];
      }
      if (this.pressureChart.data.datasets[2]) {
        this.pressureChart.data.datasets[2].label = '';
        this.pressureChart.data.datasets[2].data = [];
      }
    }
    this.pressureChart.update();
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

  private updateCostChart(iteration?: number, cost?: number): void {
    if (!this.costChart) {
      this.initCharts();
    }
    if (!this.costChart) {
      return;
    }
    if (iteration !== undefined && cost !== undefined) {
      this.iterationLabels.push(iteration.toString());
      this.costSeries.push(cost);
    }
    if (this.stageView === 'facility') {
      if (this.facilityOptimizationHistory.length) {
        const history = this.facilityOptimizationHistory;
        this.costChart.data.labels = history.map((entry) => entry.iteration.toString());
        this.costChart.data.datasets[0].label = 'Facility cost';
        this.costChart.data.datasets[0].data = history.map((entry) => entry.cost);
        if (this.costChart.data.datasets[1]) {
          this.costChart.data.datasets[1].label = '';
          this.costChart.data.datasets[1].data = [];
        }
      } else {
        const history = this.facilityTrainingHistory;
        this.costChart.data.labels = history.map((entry) => entry.epoch);
        const trainDataset = this.costChart.data.datasets[0];
        trainDataset.label = 'Facility train loss';
        trainDataset.data = history.map((entry) => entry.train_loss ?? entry.train_total_loss ?? null);
        if (!this.costChart.data.datasets[1]) {
          this.costChart.data.datasets[1] = {
            label: 'Facility val loss',
            data: [],
            borderColor: '#38bdf8',
            tension: 0.2,
          };
        }
        this.costChart.data.datasets[1].label = 'Facility val loss';
        this.costChart.data.datasets[1].data = history.map((entry) => entry.val_loss ?? null);
      }
    } else if (this.stageView === 'financial') {
      if (this.financialOptimizationHistory.length) {
        const history = this.financialOptimizationHistory;
        this.costChart.data.labels = history.map((entry) => entry.iteration.toString());
        this.costChart.data.datasets[0].label = 'Financial cost';
        this.costChart.data.datasets[0].data = history.map((entry) => entry.cost);
        if (this.costChart.data.datasets[1]) {
          this.costChart.data.datasets[1].label = '';
          this.costChart.data.datasets[1].data = [];
        }
      } else {
        const history = this.financialTrainingHistory;
        this.costChart.data.labels = history.map((entry) => entry.epoch);
        const trainDataset = this.costChart.data.datasets[0];
        trainDataset.label = 'Financial train loss';
        trainDataset.data = history.map((entry) => entry.train_loss ?? entry.train_total_loss ?? null);
        if (!this.costChart.data.datasets[1]) {
          this.costChart.data.datasets[1] = {
            label: 'Financial val loss',
            data: [],
            borderColor: '#f87171',
            tension: 0.2,
          };
        }
        this.costChart.data.datasets[1].label = 'Financial val loss';
        this.costChart.data.datasets[1].data = history.map((entry) => entry.val_loss ?? null);
      }
    } else {
      this.costChart.data.labels = this.iterationLabels;
      this.costChart.data.datasets[0].label = 'Cost';
      this.costChart.data.datasets[0].data = this.costSeries;
      if (this.costChart.data.datasets[1]) {
        this.costChart.data.datasets[1].label = '';
        this.costChart.data.datasets[1].data = [];
      }
    }
    this.costChart.update();
  }

  private updateGradientChart(iteration?: number, gradD?: number, gradV?: number): void {
    if (!this.gradChart) {
      this.initCharts();
    }
    if (!this.gradChart) {
      return;
    }
    if (iteration !== undefined) {
      this.gradLabels.push(iteration.toString());
      this.gradDSeries.push(gradD ?? null);
      this.gradVSeries.push(gradV ?? null);
    }
    if (this.stageView === 'facility') {
      if (this.facilityOptimizationHistory.length) {
        const history = this.facilityOptimizationHistory;
        this.gradChart.data.labels = history.map((entry) => entry.iteration.toString());
        this.gradChart.data.datasets[0].label = '∂J/∂P';
        this.gradChart.data.datasets[0].data = history.map((entry) => entry.gradients?.dJ_dP ?? null);
        this.gradChart.data.datasets[1].label = '∂J/∂T';
        this.gradChart.data.datasets[1].data = history.map((entry) => entry.gradients?.dJ_dT ?? null);
        if (this.gradChart.data.datasets[2]) {
          this.gradChart.data.datasets[2].label = '∂J/∂Gas';
          this.gradChart.data.datasets[2].data = history.map((entry) => entry.gradients?.dJ_dGas ?? null);
        }
      } else {
        const history = this.facilityTrainingHistory;
        this.gradChart.data.labels = history.map((entry) => entry.epoch);
        this.gradChart.data.datasets[0].label = 'Train loss';
        this.gradChart.data.datasets[0].data = history.map((entry) => entry.train_loss ?? entry.train_total_loss ?? null);
        this.gradChart.data.datasets[1].label = 'Val loss';
        this.gradChart.data.datasets[1].data = history.map((entry) => entry.val_loss ?? null);
        if (this.gradChart.data.datasets[2]) {
          this.gradChart.data.datasets[2].label = '';
          this.gradChart.data.datasets[2].data = [];
        }
      }
    } else if (this.stageView === 'financial') {
      if (this.financialOptimizationHistory.length) {
        const history = this.financialOptimizationHistory;
        this.gradChart.data.labels = history.map((entry) => entry.iteration.toString());
        this.gradChart.data.datasets[0].label = '∂J/∂Price';
        this.gradChart.data.datasets[0].data = history.map((entry) => entry.gradients?.dJ_dPrice ?? null);
        this.gradChart.data.datasets[1].label = '∂J/∂Hedge';
        this.gradChart.data.datasets[1].data = history.map((entry) => entry.gradients?.dJ_dHedge ?? null);
        if (this.gradChart.data.datasets[2]) {
          this.gradChart.data.datasets[2].label = '∂J/∂Opex';
          this.gradChart.data.datasets[2].data = history.map((entry) => entry.gradients?.dJ_dOpex ?? null);
        }
      } else {
        const history = this.financialTrainingHistory;
        this.gradChart.data.labels = history.map((entry) => entry.epoch);
        this.gradChart.data.datasets[0].label = 'Train loss';
        this.gradChart.data.datasets[0].data = history.map((entry) => entry.train_loss ?? entry.train_total_loss ?? null);
        this.gradChart.data.datasets[1].label = 'Val loss';
        this.gradChart.data.datasets[1].data = history.map((entry) => entry.val_loss ?? null);
        if (this.gradChart.data.datasets[2]) {
          this.gradChart.data.datasets[2].label = '';
          this.gradChart.data.datasets[2].data = [];
        }
      }
    } else {
      this.gradChart.data.labels = this.gradLabels;
      this.gradChart.data.datasets[0].label = '∂J/∂D';
      this.gradChart.data.datasets[0].data = this.gradDSeries;
      this.gradChart.data.datasets[1].label = '∂J/∂v';
      this.gradChart.data.datasets[1].data = this.gradVSeries;
      if (this.gradChart.data.datasets[2]) {
        this.gradChart.data.datasets[2].label = '';
        this.gradChart.data.datasets[2].data = [];
      }
    }
    this.gradChart.update();
  }

  private updateSpatialGradientChart(): void {
    if (!this.gradProfileChart) {
      this.initCharts();
    }
    if (!this.gradProfileChart) {
      return;
    }
    if (this.stageView === 'facility') {
      this.gradProfileChart.data.labels = ['Separator Pressure', 'Separator Temp', 'Gas Fraction'];
      const values = [
        this.pipelineControls.separatorPressure,
        this.pipelineControls.separatorTemp,
        this.pipelineControls.gasFraction,
      ];
      this.gradProfileChart.data.datasets[0].label = 'Facility controls';
      this.gradProfileChart.data.datasets[0].data = values;
      if (this.gradProfileChart.data.datasets[1]) {
        this.gradProfileChart.data.datasets[1].label = '';
        this.gradProfileChart.data.datasets[1].data = [];
      }
      if (this.gradProfileChart.data.datasets[2]) {
        this.gradProfileChart.data.datasets[2].label = '';
        this.gradProfileChart.data.datasets[2].data = [];
      }
    } else if (this.stageView === 'financial') {
      this.gradProfileChart.data.labels = ['Price', 'Hedge', 'Opex'];
      const values = [
        this.financialControls.pricePerUnit,
        this.financialControls.hedgeRatio,
        this.financialControls.opexMultiplier,
      ];
      this.gradProfileChart.data.datasets[0].label = 'Financial controls';
      this.gradProfileChart.data.datasets[0].data = values;
      if (this.gradProfileChart.data.datasets[1]) {
        this.gradProfileChart.data.datasets[1].label = '';
        this.gradProfileChart.data.datasets[1].data = [];
      }
      if (this.gradProfileChart.data.datasets[2]) {
        this.gradProfileChart.data.datasets[2].label = '';
        this.gradProfileChart.data.datasets[2].data = [];
      }
    } else {
      this.gradProfileChart.data.labels = this.xGrid;
      this.gradProfileChart.data.datasets[0].label = '∂u/∂x target';
      this.gradProfileChart.data.datasets[0].data = this.targetGradientProfile ?? [];
      this.gradProfileChart.data.datasets[1].label = '∂u/∂x surrogate';
      this.gradProfileChart.data.datasets[1].data = this.surrogateGradientPreview ?? [];
      this.gradProfileChart.data.datasets[2].label = '∂u/∂x optimizer';
      this.gradProfileChart.data.datasets[2].data = this.optimizerGradientProfile ?? [];
    }
    this.gradProfileChart.update();
  }

  private handleStreamMessage(message: BackendStreamMessage): void {
    if (message.type === 'iteration') {
      this.running = true;
      this.latestIteration = message.payload;
      this.optimizerProfile = message.payload.u_profile ?? [];
      if (message.payload.du_dx_profile?.length) {
        this.optimizerGradientProfile = message.payload.du_dx_profile;
      }
      this.updatePressureChart(message.payload.u_profile ?? []);
      this.updateSpatialGradientChart();
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

  private computeGradientFromProfile(profile: number[]): number[] {
    if (!this.xGrid.length || this.xGrid.length !== profile.length) {
      return [];
    }
    const gradients: number[] = [];
    for (let i = 0; i < profile.length; i += 1) {
      if (i === 0) {
        const dx = this.xGrid[1] - this.xGrid[0];
        gradients.push(dx !== 0 ? (profile[1] - profile[0]) / dx : 0);
      } else if (i === profile.length - 1) {
        const dx = this.xGrid[i] - this.xGrid[i - 1];
        gradients.push(dx !== 0 ? (profile[i] - profile[i - 1]) / dx : 0);
      } else {
        const dx = this.xGrid[i + 1] - this.xGrid[i - 1];
        gradients.push(dx !== 0 ? (profile[i + 1] - profile[i - 1]) / dx : 0);
      }
    }
    return gradients;
  }

  private handleTrainingResponse(response: SimulateResponse): void {
    this.surrogateReady = true;
    this.xGrid = response.x_grid;
    this.targetProfile = response.target_profile;
    this.targetGradientProfile = this.computeGradientFromProfile(this.targetProfile);
    this.optimizerProfile = [];
    this.optimizerGradientProfile = [];
    this.updatePressureChart();
    this.updateSpatialGradientChart();
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
    this.gradLabels = [];
    this.gradDSeries = [];
    this.gradVSeries = [];
    let lastGradientProfile: number[] = [];
    let lastPressureProfile: number[] = [];
    iterations.forEach((record) => {
      this.iterationLabels.push(record.iteration.toString());
      this.costSeries.push(record.cost);
      this.gradLabels.push(record.iteration.toString());
      this.gradDSeries.push(record.grad_D ?? null);
      this.gradVSeries.push(record.grad_V ?? null);
      if (record.u_profile?.length) {
        lastPressureProfile = record.u_profile;
      }
      if (record.du_dx_profile?.length) {
        lastGradientProfile = record.du_dx_profile;
      }
    });
    if (this.stageView === 'pipe') {
      if (this.costChart) {
        this.costChart.data.labels = this.iterationLabels;
        this.costChart.data.datasets[0].data = this.costSeries;
        if (this.costChart.data.datasets[1]) {
          this.costChart.data.datasets[1].data = [];
        }
        this.costChart.update();
      }
      if (this.gradChart) {
        this.gradChart.data.labels = this.gradLabels;
        this.gradChart.data.datasets[0].data = this.gradDSeries;
        this.gradChart.data.datasets[1].data = this.gradVSeries;
        if (this.gradChart.data.datasets[2]) {
          this.gradChart.data.datasets[2].data = [];
        }
        this.gradChart.update();
      }
    } else {
      this.updateCostChart();
      this.updateGradientChart();
    }

    if (response.result) {
      this.finalResult = response.result;
    }
    if (iterations.length) {
      this.latestIteration = iterations[iterations.length - 1];
    }
    this.optimizerProfile = lastPressureProfile;
    this.optimizerGradientProfile = lastGradientProfile;
    this.updatePressureChart();
    this.updateSpatialGradientChart();
  }

  hasGradientSamples(): boolean {
    if (this.stageView === 'facility') {
      return this.facilityOptimizationHistory.length > 0 || this.facilityTrainingHistory.length > 0;
    }
    if (this.stageView === 'financial') {
      return this.financialOptimizationHistory.length > 0 || this.financialTrainingHistory.length > 0;
    }
    return this.gradLabels.length > 0;
  }

  hasSpatialGradientSamples(): boolean {
    if (this.stageView === 'facility' || this.stageView === 'financial') {
      return true;
    }
    return (
      (this.targetGradientProfile?.length ?? 0) > 0 ||
      (this.surrogateGradientPreview?.length ?? 0) > 0 ||
      (this.optimizerGradientProfile?.length ?? 0) > 0
    );
  }

  facilityOptimizationTimestampLabel(): string | null {
    if (!this.facilityOptimizationTimestamp) {
      return null;
    }
    const parsed = Date.parse(this.facilityOptimizationTimestamp);
    if (Number.isNaN(parsed)) {
      return this.facilityOptimizationTimestamp;
    }
    return new Date(parsed).toLocaleString();
  }

  financialOptimizationTimestampLabel(): string | null {
    if (!this.financialOptimizationTimestamp) {
      return null;
    }
    const parsed = Date.parse(this.financialOptimizationTimestamp);
    if (Number.isNaN(parsed)) {
      return this.financialOptimizationTimestamp;
    }
    return new Date(parsed).toLocaleString();
  }

  showStatusOverlay(stage: 'pipe' | 'facility' | 'financial', event: MouseEvent): void {
    const target = event.currentTarget as HTMLElement;
    const targetRect = target.getBoundingClientRect();
    const parentRect = this.statusPanel?.nativeElement.getBoundingClientRect();
    const left = targetRect.left - (parentRect?.left ?? 0);
    const top = targetRect.bottom - (parentRect?.top ?? 0);
    this.statusOverlay = {
      stage,
      x: left,
      y: top + 8,
    };
  }

  hideStatusOverlay(): void {
    this.statusOverlay = null;
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
      if (ds[2]) {
        ds[2].borderColor = this.darkMode ? '#38bdf8' : '#0ea5e9';
      }
      this.gradChart.update('none');
    };

    const updateSpatialGradientColors = () => {
      if (!this.gradProfileChart) {
        return;
      }
      const ds = this.gradProfileChart.data.datasets;
      if (ds[0]) {
        ds[0].borderColor = this.darkMode ? '#f472b6' : '#ef476f';
      }
      if (ds[1]) {
        ds[1].borderColor = this.darkMode ? '#38bdf8' : '#118ab2';
      }
      if (ds[2]) {
        ds[2].borderColor = this.darkMode ? '#86efac' : '#06d6a0';
      }
      this.gradProfileChart.update('none');
    };

    applyScales(this.pressureChart);
    applyScales(this.costChart);
    applyScales(this.lossChart);
    applyScales(this.gradChart);
    applyScales(this.gradProfileChart);
    updatePressureColors();
    updateCostColors();
    updateLossColors();
    updateGradColors();
    updateSpatialGradientColors();
  }

  private updateBodyTheme(): void {
    document.body.classList.toggle('dark-theme', this.darkMode);
  }
}
