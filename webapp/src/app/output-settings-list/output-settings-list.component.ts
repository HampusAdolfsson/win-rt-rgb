import { Component, OnInit, Input } from '@angular/core';
import { IOutputSpecification } from '../models/outputspecification';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-output-settings-list',
  templateUrl: './output-settings-list.component.html',
  styleUrls: ['./output-settings-list.component.css']
})
export class OutputSettingsListComponent implements OnInit {

  specifications: IOutputSpecification[] = [];

  readonly defaultSpecification: IOutputSpecification = {
    ipAddress: '',
    numberOfLeds: 0,
    blurRadius: 1,
    saturationAdjustment: 0,
    flipHorizontally: false,
  };

  constructor(private snackBar: MatSnackBar) { }

  ngOnInit(): void {
  }

  addSpecification(): void {
    this.specifications.push(JSON.parse(JSON.stringify(this.defaultSpecification)));
  }

  applySpec(i: number): void {
    console.log(`Applying spec ${i}`);
    this.snackBar.open('Applied settings', undefined, {
      duration: 2000,
      panelClass: 'my-snack-bar',
    });
  }

  deleteSpec(i: number): void {
    this.specifications.splice(i, 1);
  }

}
