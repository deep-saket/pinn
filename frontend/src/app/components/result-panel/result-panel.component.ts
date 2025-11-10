import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';

import { IterationPayload, OptimizationResult } from '../../services/backend.service';

@Component({
  selector: 'app-result-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './result-panel.component.html',
  styleUrl: './result-panel.component.scss',
})
export class ResultPanelComponent {
  @Input() running = false;
  @Input() currentIteration?: IterationPayload;
  @Input() finalResult?: OptimizationResult;

  formatValue(value: number | undefined | null, digits = 3): string {
    if (value === undefined || value === null) {
      return '--';
    }
    return value.toFixed(digits);
  }
}
