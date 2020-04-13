import Vue from 'vue'
import Router from 'vue-router'
Vue.use(Router)
export default new Router({
  routes: [
    {
      path: '/',
      name: 'index',
      meta:{title:'法律信息综合检索平台'},
      component: ()=>import('@/view/index'),
      beforeEnter: (to, from, next) => {
        next({ path: '/dldemo' })
      }
      // beforeEnter: (to, from, next) => {
      //   console.log("--1031--"+JSON.stringify(to));
      //   if(to.query.jslx=='flfg')
      //     next({ path: '/lawsNew?q='+to.query.q })
      //   if(to.query.jslx=='sfal')
      //     next({ path: '/example?q='+to.query.q })
      //   if(to.query.jslx=='fxqk')
      //     next({ path: '/journalLaw?q='+to.query.q })
      //   else
      //     next({ path: '/oneStopSearch?q='+to.query.q })
      // }
    },
    {
      path: '/dldemo',//一站式
      name: 'dldemo',
      meta:{title:'深度学习示例'},
      component: ()=>import('@/view/dldemo')
    },
		{
		  path: '/oneStopSearch',//一站式
		  name: 'oneStopSearch',
		  meta:{title:'一站式检索'},
		  component: ()=>import('@/view/oneStopSearch')
		},
    {
      path: '/lawsNew',//法宝
      name: 'lawsNew',
      meta:{title:'法律法规'},
      component: ()=>import('@/view/lawsNew')
    },
    {
      path: '/detail/:type/:gid',
      name: 'detail1',
      meta:{title:'法律法规'},
      component: ()=>import('@/view/detail')
    },
    {
      path: '/detail/:type/:gid/:keyword',
      name: 'detail1',
      meta:{title:'法律法规'},
      component: ()=>import('@/view/detail')
    },
    {
      path: '/detail/:type/:gid/:cid',
      name: 'detail',
      meta:{title:'法律法规'},
      component: ()=>import('@/view/detail')
    },
    {
      path: '/example',
      name: 'example',
      meta:{title:'司法案例'},
      component: ()=>import('@/view/example/example')
    },
		{
		  path: '/journalLaw',
		  name: 'journalLaw',
		  meta:{title:'法学期刊'},
		  component: ()=>import('@/view/journal/journalLaw')
    },
    {
		  path: '/lawsChange',
		  name: 'lawsChange',
		  meta:{title:'法规变迁'},
		  component: ()=>import('@/view/lawsChange')
    }
  ]
})
