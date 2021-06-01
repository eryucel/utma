import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {DashboardRoutingModule} from './dashboard-routing.module';
import {DashboardComponent} from './dashboard.component';
import {DatasetComponent} from './components';
import {AddDatasetComponent} from './components';
import {SharedModule} from "../shared";
import {NgxDropzoneModule} from "ngx-dropzone";
import {EditDatasetComponent} from './components';
import {NumberAttributesStepComponent} from './components/number-attributes-step/number-attributes-step.component';
import {CategoricalAttributesStepComponent} from './components/categorical-attributes-step/categorical-attributes-step.component';
import {DataRowsStepComponent} from './components/data-rows-step/data-rows-step.component';
import {ResultComponent} from './components/result/result.component';
import {ChartsModule} from "ng2-charts";
import { TaskComponent } from './components/task/task.component';
import { ResultsComponent } from './components/results/results.component';


@NgModule({
  declarations: [
    DashboardComponent,
    DatasetComponent,
    AddDatasetComponent,
    EditDatasetComponent,
    NumberAttributesStepComponent,
    CategoricalAttributesStepComponent,
    DataRowsStepComponent,
    ResultComponent,
    TaskComponent,
    ResultsComponent
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
