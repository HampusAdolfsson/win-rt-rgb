import { Component, OnInit } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { IProfile } from '../models/profile';

@Component({
  selector: 'app-profile-list',
  templateUrl: './profile-list.component.html',
  styleUrls: ['./profile-list.component.css']
})
export class ProfileListComponent implements OnInit {

  profiles: IProfile[] = [];

  readonly defaultProfile: IProfile = {
    regex: '',
    area: {
      x: 0, y: 0,
      width: 1920, height: 1080,
    },
  };

  constructor(private snackBar: MatSnackBar) { }

  ngOnInit(): void {
  }

  addProfile(): void {
    this.profiles.push(JSON.parse(JSON.stringify(this.defaultProfile)));
  }

  saveProfile(i: number): void {
    console.log(`Saving profile ${i}`);
    this.snackBar.open('Saved profile', undefined, {
      duration: 2000,
      panelClass: 'my-snack-bar',
    });
  }

  deleteProfile(i: number): void {
    this.profiles.splice(i, 1);
  }

}
