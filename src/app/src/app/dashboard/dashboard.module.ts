import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {DashboardRoutingModule} from './dashboard-routing.module';
import {DashboardComponent} from './dashboard.component';
import {DatasetComponent} from './components';
import {AddDatasetComponent} from './components';
import {SharedModule} from "../shared";
import {MaterialModule} from "../shared/material.module";
import {NgxDropzoneModule} from "ngx-dropzone";
import { EditDatasetComponent } from './components/edit-dataset/edit-dataset.component';
import { NumberAttributesStepComponent } from './components/number-attributes-step/number-attributes-step.component';
import { CategoricalAttributesStepComponent } from './components/categorical-attributes-step/categorical-attributes-step.component';
import { DataRowsStepComponent } from './components/data-rows-step/data-rows-step.component';
import {ChartsModule} from "ng2-charts";


@NgModule({
  declarations: [
    DashboardComponent,
    DatasetComponent,
    AddDatasetComponent,
    EditDatasetComponent,
    NumberAttributesStepComponent,
    CategoricalAttributesStepComponent,
    DataRowsStepComponent
  ],
  imports: [
    CommonModule,
    DashboardRoutingModule,
    SharedModule,
    NgxDropzoneModule,
    ChartsModule
  ]
})
export class DashboardModule {
}
