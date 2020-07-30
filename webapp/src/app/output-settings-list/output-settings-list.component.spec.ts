import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { OutputSettingsListComponent } from './output-settings-list.component';

describe('OutputSettingsListComponent', () => {
  let component: OutputSettingsListComponent;
  let fixture: ComponentFixture<OutputSettingsListComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ OutputSettingsListComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(OutputSettingsListComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
