import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {FooterComponent, HeaderComponent, SpinnerComponent} from "./components";
import {RouterModule} from "@angular/router";
import {ShowAuthedDirective} from "./directives/show-authed.directive";
import {ListErrorsComponent} from './components/list-errors/list-errors.component';
import {FormsModule, ReactiveFormsModule} from "@angular/forms";
import {HttpClientModule} from "@angular/common/http";
import {MaterialModule} from "./material.module";


@NgModule({
  declarations: [
    SpinnerComponent,
    FooterComponent,
    HeaderComponent,
    ShowAuthedDirective,
    ListErrorsComponent
  ],
  exports: [
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,
    RouterModule,
    CommonModule,
    MaterialModule,
    ShowAuthedDirective,
    ListErrorsComponent,
    FooterComponent,
    HeaderComponent
  ],
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    HttpClientModule,
    RouterModule,
    MaterialModule
  ]
})
export class SharedModule {
}
