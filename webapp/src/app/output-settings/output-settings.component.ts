import { Component, OnInit } from '@angular/core';
import { IOutputSpecification } from '../models/outputspecification';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-output-settings',
  templateUrl: './output-settings.component.html',
  styleUrls: ['./output-settings.component.css']
})
export class OutputSettingsComponent implements OnInit {
  specification: IOutputSpecification;

  constructor(private snackBar: MatSnackBar) {
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
    console.log(this.specification.flipHorizontally);
    this.snackBar.open('Applied settings', undefined, {
      duration: 2000,
      panelClass: 'my-snack-bar',
    });
  }

}
