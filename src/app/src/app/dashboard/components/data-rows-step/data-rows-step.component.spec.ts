import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DataRowsStepComponent } from './data-rows-step.component';

describe('DataRowsStepComponent', () => {
  let component: DataRowsStepComponent;
  let fixture: ComponentFixture<DataRowsStepComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DataRowsStepComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DataRowsStepComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
