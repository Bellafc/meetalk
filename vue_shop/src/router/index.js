import Vue from "vue";
import VueRouter from "vue-router";
import home from "../views/home";

Vue.use(VueRouter);



const routes = [
  {
    path: "/",
    name: "home",
    component: home
  }
];

const router = new VueRouter({
  // mode: 'history',
  // base: '/test/',
  routes
});

// router.beforeEach((to, from, next) => {
//   const isLoggedIn = store.state.isLoggedIn; // 从 Vuex store 中获取登录状态
//   if (to.name !== 'login' && !isLoggedIn) {
//     next({ name: 'login' }); // 如果用户未登录且试图访问的不是登录页面，则重定向到登录页面
//   } else {
//     next(); // 允许用户继续访问其他页面
//   }
// })

export default router;
