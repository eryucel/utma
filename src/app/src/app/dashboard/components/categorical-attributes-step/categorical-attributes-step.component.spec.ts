import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CategoricalAttributesStepComponent } from './categorical-attributes-step.component';

describe('CategoricalAttributesStepComponent', () => {
  let component: CategoricalAttributesStepComponent;
  let fixture: ComponentFixture<CategoricalAttributesStepComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CategoricalAttributesStepComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CategoricalAttributesStepComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
