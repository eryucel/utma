import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {AuthRoutingModule} from './auth-routing.module';
import {AuthComponent} from './auth.component';
import {SharedModule} from "../shared";
import {ReactiveFormsModule} from "@angular/forms";
import {NoAuthGuard} from "../core";


@NgModule({
  declarations: [
    AuthComponent
  ],
  imports: [
    CommonModule,
    AuthRoutingModule,
    SharedModule,
    ReactiveFormsModule
  ],
  providers:[
    NoAuthGuard
  ]
})
export class AuthModule {
}
