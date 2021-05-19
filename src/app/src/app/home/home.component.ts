import {Component, OnInit} from '@angular/core';
import {UserService} from "../core";
import {Router} from "@angular/router";


@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {

  constructor(
    private router: Router,
    private userService: UserService) {
  }

  isAuthenticated!: boolean;

  ngOnInit(): void {
    this.userService.isAuthenticated.subscribe(
      (authenticated) => {
        this.isAuthenticated = authenticated;

        // set the article list accordingly
        if (authenticated) {
        } else {
        }
      }
    );

    // this.tagsService.getAll()
    //   .subscribe(tags => {
    //     this.tags = tags;
    //     this.tagsLoaded = true;
    //   });
  }

}
