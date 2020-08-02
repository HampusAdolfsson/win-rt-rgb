import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { IProfile } from '../models/profile';

@Component({
  selector: 'app-profile',
  templateUrl: './profile.component.html',
  styleUrls: ['./profile.component.css']
})
export class ProfileComponent implements OnInit {
  @Input() profile: IProfile;
  @Input() isLocked: IProfile;
  @Output() saved = new EventEmitter<void>();
  @Output() deleted = new EventEmitter<void>();
  @Output() locked = new EventEmitter<boolean>();

  constructor() { }

  ngOnInit(): void {
  }

  save(): void {
    this.saved.emit();
  }

  lock(): void {
    this.locked.emit(!this.isLocked);
  }

  delete(): void {
    this.deleted.emit();
  }

}
