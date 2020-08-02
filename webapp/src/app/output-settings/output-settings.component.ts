import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { IOutputSpecification } from '../models/outputspecification';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-output-settings',
  templateUrl: './output-settings.component.html',
  styleUrls: ['./output-settings.component.css']
})
export class OutputSettingsComponent implements OnInit {
  @Input() specification: IOutputSpecification;
  @Output() applied = new EventEmitter<IOutputSpecification>();
  @Output() deleted = new EventEmitter<IOutputSpecification>();

  constructor() {
    if (!false) {
      this.specification = {
        ipAddress: '',
        numberOfLeds: 0,
        blurRadius: 1,
        saturationAdjustment: 0,
        flipHorizontally: false,
      };
    } else {
      // this.specification = spec;
    }
  }

  ngOnInit(): void {
  }

  apply(): void {
    this.applied.emit(this.specification);
  }

  delete(): void {
    this.deleted.emit(this.specification);
  }

}
