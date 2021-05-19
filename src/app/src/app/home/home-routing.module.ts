import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {HomeAuthResolver} from "./home-auth-resolver.service";
import {HomeComponent} from "./home.component";

const routes: Routes = [
  {
    path: '',
    component: HomeComponent,
    resolve: {
      isAuthenticated: HomeAuthResolver
    }
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class HomeRoutingModule { }
