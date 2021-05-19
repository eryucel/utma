import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { HomeRoutingModule } from './home-routing.module';
import { HomeComponent } from './home.component';
import {SharedModule} from "../shared";
import {HomeAuthResolver} from "./home-auth-resolver.service";


@NgModule({
  declarations: [
    HomeComponent
  ],
  imports: [
    CommonModule,
    HomeRoutingModule,
    SharedModule
  ],
  providers: [
    HomeAuthResolver
  ]
})
export class HomeModule { }
