import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NumberAttributesStepComponent } from './number-attributes-step.component';

describe('NumberAttributesStepComponent', () => {
  let component: NumberAttributesStepComponent;
  let fixture: ComponentFixture<NumberAttributesStepComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ NumberAttributesStepComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(NumberAttributesStepComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
