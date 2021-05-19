import { ComponentFixture, TestBed } from '@angular/core/testing';

import { EditDatasetComponent } from './edit-dataset.component';

describe('EditDatasetComponent', () => {
  let component: EditDatasetComponent;
  let fixture: ComponentFixture<EditDatasetComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ EditDatasetComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(EditDatasetComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
